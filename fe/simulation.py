import numpy as np
import time

import os
from rdkit import Chem

from simtk.openmm import app
from simtk.openmm.app import forcefield as ff
from simtk.openmm.app import PDBFile

from openforcefield.typing.engines import smirnoff
from system import serialize, forcefield

from timemachine.lib import custom_ops
import traceback


def merge_gradients(
    gradients,
    precision):

    g = []
    for fn, fn_args in gradients:
        g.append(fn(*fn_args, precision=precision))

    return g

class Simulation:
    """
    A picklable simulation object
    """
    def __init__(self,
        guest_mol,
        host_pdb,
        direction,
        step_sizes,
        cas,
        lambda_schedule,
        perm
        ):

        self.step_sizes = step_sizes
        amber_ff = app.ForceField('amber99sb.xml', 'tip3p.xml')

        # host
        system = amber_ff.createSystem(
            host_pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            rigidWater=False)

        host_potentials, (host_params, host_param_groups), host_masses = serialize.deserialize_system(system, dimension=4)
        # parameterize the small molecule
        off = smirnoff.ForceField("test_forcefields/smirnoff99Frosst.offxml")
        guest_potentials, (guest_params, guest_param_groups), guest_masses = forcefield.parameterize(guest_mol, off, dimension=4)

        combined_potentials, combined_params, combined_param_groups, combined_masses = forcefield.combiner(
            host_potentials, guest_potentials,
            host_params, guest_params,
            host_param_groups, guest_param_groups,
            host_masses, guest_masses,
            perm
        )

        self.num_host_atoms = len(host_masses)

        self.perm = perm
        self.iperm = np.argsort(perm)
        self.combined_potentials = combined_potentials
        self.combined_params = combined_params
        self.combined_param_groups = combined_param_groups
        self.combined_masses = combined_masses[perm]

        N_host = len(host_pdb.positions)
        N_guest = guest_mol.GetNumAtoms()
        N_combined = N_host + N_guest

        self.cas = cas
        self.cbs = -np.ones(N_combined)*0.001

        self.lambda_schedule = lambda_schedule
        self.lambda_idxs = np.zeros(N_combined, dtype=np.int32)
        if direction == 'deletion':
            self.lambda_idxs[N_host:] = 1
        elif direction == 'insertion':
            self.lambda_idxs[N_host:] = -1
        else:
            raise ValueError("Unknown direction: "+direction)
        # how did this take 6 hours to debug?
        self.lambda_idxs = self.lambda_idxs[perm]
        self.exponent = 16

    def run_forward_multi(self, args):
        x0, pdb_writer, gpu_idx, precision = args
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
        try:
            return self.run_forward(x0, pdb_writer, precision)
        except Exception as err:
            print(err)
            traceback.print_tb(err.__traceback__)
            raise 

    def run_forward(self, x0, pdb_writer, precision):
        """
        x0 include host configs as well
        """
        # this is multi-process safe to run.
        # use single precision
        gradients = merge_gradients(self.combined_potentials, precision)

        stepper = custom_ops.LambdaStepper_f64(
            gradients,
            self.lambda_schedule,
            self.lambda_idxs,
            self.exponent
        )

        v0 = np.zeros_like(x0)

        ctxt = custom_ops.ReversibleContext_f64_3d(
            stepper,
            x0,
            v0,
            self.cas,
            self.cbs,
            self.step_sizes,
            self.combined_params
        )
        start = time.time()
        ctxt.forward_mode()
        print("run time", time.time() - start)
        du_dls = stepper.get_du_dl()

        if pdb_writer is not None:
            pdb_writer.write_header()
            xs = ctxt.get_all_coords()
            for frame_idx, x in enumerate(xs):

                interval = max(1, xs.shape[0]//pdb_writer.n_frames)
                if frame_idx % interval == 0:
                    # argsort is iperm and
                    pdb_writer.write((x*10)[self.iperm])
        # pdb_writer.close()
        del stepper
        del ctxt

        return du_dls

    def run_forward_and_backward_multi(self, args):
        x0, du_dl_adjoints, gpu_idx = args
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
        try:
            return self.run_forward_and_backward(x0)
        except Exception as e:
            print(e)
            traceback.print_tb(err.__traceback__)
            raise 


    def run_forward_and_backward(self, x0, du_dl_adjoints):
        """
        x0 include host configs as well
        """

        start = time.time()
        gradients = []
        for fn, fn_args in self.combined_potentials:
            gradients.append(fn(*fn_args))



        stepper = custom_ops.LambdaStepper_f64(
            gradients,
            self.lambda_schedule,
            self.lambda_idxs,
            self.exponent
        )

        v0 = np.zeros_like(x0)

        ctxt = custom_ops.ReversibleContext_f64_3d(
            stepper,
            x0,
            v0,
            self.cas,
            self.cbs,
            self.step_sizes,
            self.combined_params,
        )
        ctxt.forward_mode()

        stepper.set_du_dl_adjoint(dloss_ddudl)
        ctxt.set_x_t_adjoint(np.zeros_like(x0))
        ctxt.backward_mode()

        dL_dp = ctxt.get_param_adjoint_accum()

        print("run time", time.time() - start)

        return dL_dp
