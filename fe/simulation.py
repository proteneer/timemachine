import numpy as np
import time

import os
from rdkit import Chem

from simtk.openmm import app

from timemachine.lib import ops, custom_ops
import traceback


import jax
import jax.numpy as jnp


# import warnings
# warnings.simplefilter("ignore", UserWarning)


def check_coords(x):
    x3 = x[:, :3]
    ri = np.expand_dims(x3, 0)
    rj = np.expand_dims(x3, 1)
    dij = np.linalg.norm(ri-rj, axis=-1)

    N = x.shape[0]

    idx = np.argmax(dij)
    row = idx // N
    col = idx - row * N 

    if np.any(dij > 100):
        return False
    elif np.any(np.isinf(dij)):
        return False
    elif np.any(np.isnan(dij)):
        return False

    return True

# check_coords = jax.jit(check_coords, static_argnums=(0,))

class Simulation:
    """
    A serializable simulation object
    """
    def __init__(self,
        lhs_system,
        rhs_system,
        step_sizes,
        cas,
        cbs,
        ccs,
        lambda_schedule,
        precision):
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

        precision: either np.float64 or np.float32
            Precision in which we compute the force kernels. Note that integration
            is always done in 64bit.

        """

        self.step_sizes = step_sizes
        self.lhs_system = lhs_system
        self.rhs_system = rhs_system
        self.cas = cas
        self.cbs = cbs
        self.ccs = ccs

        for b in cbs:
            if b > 0:
                raise ValueError("cbs must all be <= 0")

        self.lambda_schedule = lambda_schedule
        self.precision = precision

    def run_forward_and_backward(
        self,
        x0,
        v0,
        params,
        seed,
        pdb_writer,
        pipe,
        gpu_idx):
        """
        Run a forward simulation

        Parameters
        ----------
        x0: np.arrray, np.float64, [N,3]
            Starting geometries

        v0: np.array, np.float64, [N, 3]
            Starting velocities

        params: np.array, np.float64, [P]
            Epoch parameters

        seed: int
            Random number used to seed the thermostat

        pdb_writer: For writing out the trajectory
            If None then we skip writing

        pipe: multiprocessing.Pipe
            Use to communicate with the parent host

        gpu_idx: int
            which gpu we run the job on


        The pipe will ping-pong in two passes. If the simulation is stable, ie. the coords
        of the last frame is well defined, then we return du_dls. Otherwise, a None is sent
        through the pipe. The pipe then expects to receive the adjoint derivatives, which
        must be sent for the function to return. None adjoints can be sent to instantly
        return the function.

        """
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)

        gradients = []
        handles = []
        force_names = []


        for k, v in self.lhs_system.nrg_fns.items():

            force_names.append(k)
            other_v = self.rhs_system.nrg_fns[k]
            op_fn = getattr(ops, k)
            grad = op_fn(*v, precision=self.precision)
            grad_other = op_fn(*other_v, precision=self.precision)
            handles.append(grad)
            handles.append(grad_other)
            grad_alchem = ops.AlchemicalGradient(
                len(self.lhs_system.masses),
                len(self.lhs_system.params),
                grad,
                grad_other
            )
            gradients.append(grad_alchem)

        # x_bad = np.load("all_coords.npy")[3497]

        # for g in gradients:
        #     forces, du_dl, energy = g.execute_lambda(x_bad, self.lhs_system.params, 0.20)
        #     print(g, forces[1758+14], forces[1758+51], np.amax(np.abs(forces[1758:])), np.argmax(np.abs(forces[1758:]), axis=0))

        # assert 0

        stepper = custom_ops.AlchemicalStepper_f64(
            gradients,
            self.lambda_schedule
        )

        v0 = np.zeros_like(x0)

        np.testing.assert_equal(self.lhs_system.params, self.rhs_system.params)
        ctxt = custom_ops.ReversibleContext_f64(
            stepper,
            x0,
            v0,
            self.cas,
            self.cbs,
            self.ccs,
            self.step_sizes,
            params,
            seed
        )

        start = time.time()
        # print("start_forward_mode")
        ctxt.forward_mode()
        print("fwd run time", time.time() - start)

        xs = ctxt.get_all_coords()

        np.save("all_coords.npy", xs)
        # # np.save("debug_coords.npy", xs[6000])

        # assert 0


        start = time.time()
        x_final = ctxt.get_last_coords()[:, :3]

        energies = stepper.get_energies()
        for e_idx, e in enumerate(energies):
            if e_idx > 50:
                break

        if check_coords(x_final) == False:
            print("FATAL WARNING: ------ Final frame FAILED ------")
            du_dls = None

        full_du_dls = stepper.get_du_dl()


        print("Max nonbonded arg", np.argmax(full_du_dls[3]))

        full_energies = stepper.get_energies()

        # equil_du_dls = full_du_dls
        equil_du_dls = full_du_dls[:, len(self.step_sizes)//2:]

        # print(equil_du_dls.shape)

        # assert 0

        for fname, du_dls in zip(force_names, equil_du_dls):
            print("lambda:", "{:.2f}".format(self.lambda_schedule[0]), "\t mean/std du_dls", "{:8.2f}".format(np.mean(du_dls)), "+-", "{:7.2f}".format(np.std(du_dls)), "\t <-", fname)
            # print("lambda:", "{:.2f}".format(self.lambda_schedule[0]), "\t mean/std du_dls", "{:8.2f}".format(np.trapz(du_dls, self.lambda_schedule)), "+-", "{:7.2f}".format(np.std(du_dls)), "\t <-", fname)

        total_equil_du_dls = np.sum(equil_du_dls, axis=0)

        print("lambda:", "{:.2f}".format(self.lambda_schedule[0]), "\t mean/std du_dls", "{:8.2f}".format(np.mean(total_equil_du_dls)), "+-", "{:7.2f}".format(np.std(total_equil_du_dls)), "\t <- Total")
        # print("lambda:", "{:.2f}".format(self.lambda_schedule[0]), "\t mean/std du_dls", "{:8.2f}".format(np.trapz(total_equil_du_dls, self.lambda_schedule)), "+-", "{:7.2f}".format(np.std(total_equil_du_dls)), "\t <- Total")

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

        pipe.send((full_du_dls, full_energies))

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
