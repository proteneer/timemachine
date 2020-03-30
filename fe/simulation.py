import numpy as np
import time

import os
from rdkit import Chem

from simtk.openmm import app

from timemachine.lib import custom_ops
import traceback


import jax
import jax.numpy as jnp


import warnings
warnings.simplefilter("ignore", UserWarning)

def check_coords(x):
    x3 = x[:, :3]
    ri = jnp.expand_dims(x3, 0)
    rj = jnp.expand_dims(x3, 1)
    dij = jnp.linalg.norm(ri-rj, axis=-1)

    N = x.shape[0]

    idx = jnp.argmax(dij)
    row = idx // N
    col = idx - row * N 

    if jnp.any(dij > 100):
        return False
    elif jnp.any(jnp.isinf(dij)):
        return False
    elif jnp.any(jnp.isnan(dij)):
        return False

    return True

check_coords = jax.jit(check_coords, static_argnums=(0,))


class Simulation:
    """
    A serializable simulation object
    """
    def __init__(self,
        system,
        step_sizes,
        cas,
        cbs,
        ccs,
        lambda_schedule,
        lambda_idxs,
        precision,
        seed):
        """
        Create a simulation.

        Parameters
        ----------
        system: System
            A fully parameterized system

        step_sizes: np.array, np.float64, [T]
            dt for each step

        cas: np.array, np.float64, [T]
            Friction coefficient to be used on each timestep

        cbs: np.array, np.float64, [N]
            Per particle force multipliers. Every element must be negative.

        lambda_schedule: np.array, np.float64, [T]
            lambda parameter for each time step

        lambda_idxs: np.array, np.int32, [N]
            If lambda_idxs[atom_idx] is 0, then the particle is not modified.

        precision: either np.float64 or np.float32
            Precision in which we compute the force kernels. Note that integration
            is always done in 64bit.

        """

        self.step_sizes = step_sizes
        self.system = system
        self.cas = cas
        self.cbs = cbs
        self.ccs = ccs
        self.seed = seed

        for b in cbs:
            if b > 0:
                raise ValueError("cbs must all be <= 0")

        self.lambda_schedule = lambda_schedule
        self.lambda_idxs = lambda_idxs
        self.precision = precision

    def run_forward_and_backward(
        self,
        x0,
        v0,
        gpu_idx,
        pdb_writer,
        pipe):
        """
        Run a forward simulation

        Parameters
        ----------
        x0: np.arrray, np.float64, [N,3]
            Starting geometries

        v0: np.array, np.float64, [N, 3]
            Starting velocities

        gpu_idx: int
            which gpu we run the job on

        pdb_writer: For writing out the trajectory
            If None then we skip writing

        pipe: multiprocessing.Pipe
            Use to communicate with the parent host

        The pipe will ping-pong in two passes. If the simulation is stable, ie. the coords
        of the last frame is well defined, then we return du_dls. Otherwise, a None is sent
        through the pipe. The pipe then expects to receive the adjoint derivatives, which
        must be sent for the function to return. None adjoints can be sent to instantly
        return the function.

        """
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)

        gradients = self.system.make_gradients(dimension=4, precision=self.precision)

        # (ytz): debug use
        # gradients = self.system.make_gradients(dimension=3, precision=self.precision)
        # for g in gradients:
        #     forces = g.execute(x0, self.system.params)
        #     print(g, forces, np.amax(np.abs(forces)))

        # assert 0

        stepper = custom_ops.LambdaStepper_f64(
            gradients,
            self.lambda_schedule,
            self.lambda_idxs
        )

        v0 = np.zeros_like(x0)
        ctxt = custom_ops.ReversibleContext_f64_3d(
            stepper,
            x0,
            v0,
            self.cas,
            self.cbs,
            self.ccs,
            self.step_sizes,
            self.system.params,
            self.seed
        )

        start = time.time()
        print("start_forward_mode")
        ctxt.forward_mode()

        print("fwd run time", time.time() - start)

        du_dls = stepper.get_du_dl()

        start = time.time()
        x_final = ctxt.get_last_coords()[:, :3]

        if check_coords(x_final) == False:
            print("Final frame failed")
            du_dls = None

        if pdb_writer is not None:
            pdb_writer.write_header()
            xs = ctxt.get_all_coords()
            for frame_idx, x in enumerate(xs):
                interval = max(1, xs.shape[0]//pdb_writer.n_frames)
                if frame_idx % interval == 0:
                    if check_coords(x):
                        pdb_writer.write(x*10)
                    else:
                        print("failed to write on frame", frame_idx)
                        break
            pdb_writer.close()

        print("sending dudls back.")
        pipe.send(du_dls)

        du_dl_adjoints = pipe.recv()

        if du_dl_adjoints is not None:
            stepper.set_du_dl_adjoint(du_dl_adjoints)
            ctxt.set_x_t_adjoint(np.zeros_like(x0))
            start = time.time()
            ctxt.backward_mode()
            print("bkwd run time", time.time() - start)
            dL_dp = ctxt.get_param_adjoint_accum()
            pipe.send(dL_dp)

        pipe.close()
        return
