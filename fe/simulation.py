import numpy as np
import time


from rdkit import Chem

from simtk.openmm import app
from simtk.openmm.app import forcefield as ff
from simtk.openmm.app import PDBFile

from openforcefield.typing.engines import smirnoff
from system import serialize, forcefield

from timemachine.lib import custom_ops

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
            host_masses, guest_masses)

        self.combined_potentials = combined_potentials
        self.combined_params = combined_params
        self.combined_param_groups = combined_param_groups
        self.combined_masses = combined_masses



        N_host = len(host_pdb.positions)
        N_guest = guest_mol.GetNumAtoms()
        N_combined = N_host + N_guest

        self.cas = cas
        self.cbs = -np.ones(N_combined)*0.001

        self.lambda_schedule = lambda_schedule
        self.lambda_idxs = np.zeros(N_combined, dtype=np.int32)
        if direction == 'deletion':
            self.lambda_idxs[N_host:] = 1 # insertion is -1, deletion is +1
        elif direction == 'insertion':
            self.lambda_idxs[N_host:] = -1 # insertion is -1, deletion is +1
        else:
            raise ValueError("Unknown direction: "+direction)
        self.exponent = 1

    def run_forward(self, x0):
        """
        x0 include host configs as well
        """
        # this is multi-process safe to run.

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
            len(self.combined_masses),
            x0.reshape(-1).tolist(),
            v0.reshape(-1).tolist(),
            self.cas.tolist(),
            self.cbs.tolist(),
            self.step_sizes.tolist(),
            self.combined_params.reshape(-1).tolist(),
        )
        print("init time", time.time() - start)

        start = time.time()
        ctxt.forward_mode()
        # print("run time", time.time() - start)

        du_dls = stepper.get_du_dl()

        stepper.set_du_dl_adjoint(np.zeros_like(du_dls))

        # test_adjoint = np.random.rand(x0.shape[0], x0.shape[0])/10
        ctxt.set_x_t_adjoint(np.zeros_like(x0))

        ctxt.backward_mode()

        print("run time", time.time() - start)

        del stepper
        del ctxt

        return du_dls


    # def run_forward(self, x0):
    #     """
    #     x0 include host configs as well
    #     """
    #     # this is multi-process safe to run.

    #     start = time.time()
    #     gradients = []
    #     for fn, fn_args in self.combined_potentials:
    #         gradients.append(fn(*fn_args))

    #     stepper = custom_ops.LambdaStepper_f64(
    #         gradients,
    #         self.lambda_schedule,
    #         self.lambda_idxs,
    #         self.exponent
    #     )

    #     v0 = np.zeros_like(x0)

    #     ctxt = custom_ops.ReversibleContext_f64_3d(
    #         stepper,
    #         len(self.combined_masses),
    #         x0.reshape(-1).tolist(),
    #         v0.reshape(-1).tolist(),
    #         self.cas.tolist(),
    #         self.cbs.tolist(),
    #         self.step_sizes.tolist(),
    #         self.combined_params.reshape(-1).tolist(),
    #     )
    #     print("init time", time.time() - start)

    #     start = time.time()
    #     ctxt.forward_mode()


    #     du_dls = stepper.get_du_dl()

    #     ctxt.backward_mode()

    #     print("run time", time.time() - start)

    #     del stepper
    #     del ctxt

    #     return du_dls