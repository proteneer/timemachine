import numpy as np
import time

import os
from rdkit import Chem

from simtk.openmm import app

from timemachine.lib import ops, custom_ops
import traceback


import jax
import jax.numpy as jnp


from timemachine.potentials import jax_utils

# import warnings
# warnings.simplefilter("ignore", UserWarning)


def setup_harmonic_core_restraints(conf, nha, core_atoms, params):
    ri = np.expand_dims(conf, axis=0)
    rj = np.expand_dims(conf, axis=1)
    dij = jax_utils.distance(ri, rj)
    all_nbs = []

    # bond_params = []
    bond_param_idxs = []
    bond_idxs = []


    print("START PARAMS LENGTH", params.shape)

    for l_idx, dists in enumerate(dij[nha:]):
        if l_idx in core_atoms:

            nns = np.argsort(dists[:nha])
            for p_idx in nns[:10]:
                # p_idx = np.argmin(dists[:nha])
                k = 1000.0
                k_idx = len(params)
                params = np.concatenate([params, [k]])

                b = dists[p_idx]
                b_idx = len(params)
                params = np.concatenate([params, [b]])

                bond_param_idxs.append([k_idx, b_idx])
                bond_idxs.append([l_idx + nha, p_idx])

    print(np.array(bond_idxs, dtype=np.int32))
    print(np.array(bond_param_idxs, dtype=np.int32))

    print("CORE RESTRAINTS", bond_idxs)

    print("END PARAMS LENGTH", params.shape)

    return ops.HarmonicBond(
        np.array(bond_idxs, dtype=np.int32),
        np.array(bond_param_idxs, dtype=np.int32),
        precision=np.float32), params



def find_pocket_neighbors(conf, n_host, cutoff=0.5):
    """
    Find all protein atoms that we within cutoff of a ligand atom.
    """
    ri = np.expand_dims(conf, axis=0)
    rj = np.expand_dims(conf, axis=1)
    dij = jax_utils.distance(ri, rj)
    all_nbs = []
    for l_idx, dists in enumerate(dij[n_host:]):
        nbs = np.argwhere(dists[:n_host] < cutoff)
        all_nbs.extend(nbs.reshape(-1).tolist())

    return list(set(all_nbs))


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
        system,
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
        self.system = system
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
        force_names = []
        handles = []

        nha = 1758

        for k, v in self.system.nrg_fns.items():

            force_names.append(k)
            op_fn = getattr(ops, k)
            grad = op_fn(*v, precision=self.precision)
            gradients.append(grad)


        pocket_atoms = find_pocket_neighbors(x0, nha)
        core_atoms = [4,5,6,7,8,9,10,11,12,13,15,16,18]
        core_restraints, new_params = setup_harmonic_core_restraints(x0, nha, core_atoms, self.system.params)
        force_names.append("CoreRestraints")
        gradients.append(core_restraints)

        # x_bad = np.load("all_coords.npy")[1471]
        # print('--')
        # for g in gradients:
        #     forces, du_dl, energy = g.execute_lambda(x_bad, self.system.params, 0.0)
        #     norms = np.linalg.norm(forces, axis=1)
        #     highest_forces = np.argsort(norms)[::-1][:5]
        #     print(norms.shape, highest_forces)
        #     print(forces[highest_forces])

        #     print(g, np.amax(np.abs(forces)), du_dl, energy)

        # assert 0

        stepper = custom_ops.AlchemicalStepper_f64(
            gradients,
            self.lambda_schedule
        )

        v0 = np.zeros_like(x0)

        ctxt = custom_ops.ReversibleContext_f64(
            stepper,
            x0,
            v0,
            self.cas,
            self.cbs,
            self.ccs,
            self.step_sizes,
            new_params,
            seed
        )

        print("WTF PARAMS?", params.shape)

        start = time.time()
        print("start_forward_mode")
        try:
            ctxt.forward_mode()
        except e:
            print(e)

        print("fwd run time", time.time() - start)

        # xs = ctxt.get_all_coords()


        # # np.save("debug_coords.npy", xs[6000])

        # assert 0
        start = time.time()
        x_final = ctxt.get_last_coords()[:, :3]

        # energies = stepper.get_energies()
        # for e_idx, e in enumerate(energies):
        #     print(e_idx, e)
        #     if e_idx > 1600:
        #         break

        xs = ctxt.get_all_coords()

        # np.save("all_coords.npy", xs)

        # for g in gradients:
        #     forces, du_dl, energy = g.execute_lambda(xs[1460], self.system.params, 0.0)
        #     print(g, np.amax(np.abs(forces)), du_dl, energy)

        # print("--")

        # for g in gradients:
        #     forces, du_dl, energy = g.execute_lambda(xs[1469], self.system.params, 0.0)
        #     print(g, np.amax(np.abs(forces)), du_dl, energy)

        # print("--")
        # for g in gradients:
        #     forces, du_dl, energy = g.execute_lambda(xs[1470], self.system.params, 0.0)
        #     print(g, np.amax(np.abs(forces)), du_dl, energy)

        # print("--")
        # for g in gradients:
        #     forces, du_dl, energy = g.execute_lambda(xs[1471], self.system.params, 0.0)
        #     print(g, np.amax(np.abs(forces)), du_dl, energy)

        # print("--")
        # for g in gradients:
        #     forces, du_dl, energy = g.execute_lambda(xs[1472], self.system.params, 0.0)
        #     print(g, np.amax(np.abs(forces)), du_dl, energy)

        # assert 0

        if check_coords(x_final) == False:
            print("FATAL WARNING: ------ Final frame FAILED ------")
            du_dls = None

        full_du_dls = stepper.get_du_dl()


        print("Max nonbonded arg", np.argmin(full_du_dls[3]))

        full_energies = stepper.get_energies()

        # equil_du_dls = full_du_dls
        equil_du_dls = full_du_dls[:, 20000:]

        # print(equil_du_dls.shape)

        # assert 0

        for fname, du_dls in zip(force_names, equil_du_dls):
            print("lambda:", "{:.3f}".format(self.lambda_schedule[0]), "\t median {:8.2f}".format(np.median(du_dls)), "\t mean/std du_dls", "{:8.2f}".format(np.mean(du_dls)), "+-", "{:7.2f}".format(np.std(du_dls)), "\t <-", fname)
            # print("lambda:", "{:.2f}".format(self.lambda_schedule[0]), "\t mean/std du_dls", "{:8.2f}".format(np.trapz(du_dls, self.lambda_schedule)), "+-", "{:7.2f}".format(np.std(du_dls)), "\t <-", fname)

        total_equil_du_dls = np.sum(equil_du_dls, axis=0)

        print("lambda:", "{:.3f}".format(self.lambda_schedule[0]), "\t mean/std du_dls", "{:8.2f}".format(np.mean(total_equil_du_dls)), "+-", "{:7.2f}".format(np.std(total_equil_du_dls)), "\t <- Total")
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
