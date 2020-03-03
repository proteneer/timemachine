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
        lambda_schedule
        ):

        self.step_sizes = step_sizes
        amber_ff = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')

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
            host_masses, guest_masses
        )

        self.num_host_atoms = len(host_masses)

        self.combined_potentials = combined_potentials
        self.combined_params = combined_params
        self.combined_param_groups = combined_param_groups

        N_host = len(host_pdb.positions)
        N_guest = guest_mol.GetNumAtoms()
        N_combined = N_host + N_guest

        self.cas = cas
        self.cbs = -np.ones(N_combined)*0.0001

        self.lambda_schedule = lambda_schedule
        self.lambda_idxs = np.zeros(N_combined, dtype=np.int32)
        if direction == 'deletion':
            self.lambda_idxs[N_host:] = 1
        elif direction == 'insertion':
            self.lambda_idxs[N_host:] = -1
        else:
            raise ValueError("Unknown direction: "+direction)

    def run_forward_multi(self, args):
        """
        A multiprocess safe version of the run_forward code. This code will
        also set the GPU on which the simulation should run on.
        """

        x0, pdb_writer, gpu_idx, precision, adjoint_du_dl = args
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
        try:
            return self.run_forward(x0, pdb_writer, precision, adjoint_du_dl)
        except Exception as err:
            print(err)
            traceback.print_tb(err.__traceback__)
            raise 

    def run_forward(self, x0, pdb_writer, precision, du_dl_adjoints=None):
        """
        Run a forward simulation

        Parameters
        ----------
        x0: np.arrray, np.float64, [N,3]
            Starting geometries

        pdb_writer: For writing out the trajectory
            If None then we skip writing

        precision: np.float64 or np.float32
            What level of precision we run the simulation at.

        du_dl_adjoints: np.array, np.float64, [T]
            If None, then we skip the backwards pass. If not None, this
            array must have shape equal to the number of timesteps.

        """
        gradients = merge_gradients(self.combined_potentials, precision)

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
            self.step_sizes,
            self.combined_params
        )

        start = time.time()
        ctxt.forward_mode()
        print("fwd run time", time.time() - start)

        if du_dl_adjoints is not None:

            assert du_dl_adjoints.shape == self.lambda_schedule.shape
            stepper.set_du_dl_adjoint(du_dl_adjoints)
            ctxt.set_x_t_adjoint(np.zeros_like(x0))
            start = time.time()
            ctxt.backward_mode()
            print("bkwd run time", time.time() - start)
            dL_dp = ctxt.get_param_adjoint_accum()

            return dL_dp                 

        else:


            if pdb_writer is not None:
                pdb_writer.write_header()
                xs = ctxt.get_all_coords()
                for frame_idx, x in enumerate(xs):

                    interval = max(1, xs.shape[0]//pdb_writer.n_frames)
                    if frame_idx % interval == 0:
                        pdb_writer.write(x*10)
            pdb_writer.close()
            
            du_dls = stepper.get_du_dl()

            return du_dls     
