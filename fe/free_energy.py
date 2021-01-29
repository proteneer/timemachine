from jax.config import config; config.update("jax_enable_x64", True)

import jax
import numpy as np

from fe import topology

from timemachine.lib import potentials, custom_ops
from timemachine.lib import LangevinIntegrator

from ff.handlers import openmm_deserializer

def get_romol_conf(mol):
    """Coordinates of mol's 0th conformer, in nanometers"""
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf/10 # from angstroms to nm

class BaseFreeEnergy():

    @staticmethod
    def _process_grads(grads, final_vjp_and_handles):
        """
        Backpropagate system parameter derivatives into forcefield parameter derivatives.

        Parameters
        ----------
        grads: list of np array
            System derivatives in the same order as final_vjp_and_handles

        final_vjp_and_handles: list of 2-tuples
            Each 2-tuple is a (vjp_fn, handlers)


        Returns
        -------
        list of 2-tuples
            Each 2-tuple is a (np.array, class) where class indicates
            the type of the handler.

        """

        grads_and_handles = []

        assert len(grads) == len(final_vjp_and_handles)

        for du_dqs, vjps_and_handles in zip(grads, final_vjp_and_handles):

            vjp_fn = vjps_and_handles[0]
            handles = vjps_and_handles[1]

            # (ytz): nonbonded and torsion handlers map back out to multiple handlers.
            if type(handles) == tuple:
                du_dps = vjp_fn(du_dqs)
                for du_dp, handler in zip(du_dps, handles):
                    grads_and_handles.append((du_dp, type(handler)))
            else:
                du_dp = vjp_fn(du_dqs)
                # bonded terms return a list, so we need to flatten it here
                grads_and_handles.append((du_dp[0], type(handles)))

        return grads_and_handles

    @staticmethod
    def _get_integrator(combined_masses):
        """
        Get a integrator. The resulting impl must be bound to a python handle
        whose lifetime is concurrent with that of the context.
        """
        seed = np.random.randint(np.iinfo(np.int32).max)

        return LangevinIntegrator(
            300.0,
            1.5e-3,
            1.0,
            combined_masses,
            seed
        )

    # this will be eventually gRPC'd out to a worker
    @staticmethod
    def _simulate(lamb, box, x0, v0, final_potentials, integrator, equil_steps, prod_steps):
        all_impls = []
        bonded_impls = []
        nonbonded_impls = []

        # set up observables for du_dps here as well.

        du_dp_obs = []

        for bp in final_potentials:

            impl = bp.bound_impl(np.float32)

            if isinstance(bp, potentials.InterpolatedPotential) or isinstance(bp, potentials.LambdaPotential):
                bp = bp.get_u_fn()

            if isinstance(bp, potentials.Nonbonded):
                nonbonded_impls.append(impl)
            else:
                bonded_impls.append(impl)

            all_impls.append(impl)
            du_dp_obs.append(custom_ops.AvgPartialUPartialParam(impl, 5))

        intg_impl = integrator.impl()
        # context components: positions, velocities, box, integrator, energy fxns
        ctxt = custom_ops.Context(
            x0,
            v0,
            box,
            intg_impl,
            all_impls
        )

        # equilibration
        for step in range(equil_steps):
            ctxt.step(lamb)

        bonded_du_dl_obs = custom_ops.FullPartialUPartialLambda(bonded_impls, 5)
        nonbonded_du_dl_obs = custom_ops.FullPartialUPartialLambda(nonbonded_impls, 5)

        # add observable
        ctxt.add_observable(bonded_du_dl_obs)
        ctxt.add_observable(nonbonded_du_dl_obs)

        for obs in du_dp_obs:
            ctxt.add_observable(obs)

        for _ in range(prod_steps):
            ctxt.step(lamb)

        bonded_full_du_dls = bonded_du_dl_obs.full_du_dl()
        nonbonded_full_du_dls = nonbonded_du_dl_obs.full_du_dl()

        bonded_mean, bonded_std = np.mean(bonded_full_du_dls), np.std(bonded_full_du_dls)
        nonbonded_mean, nonbonded_std = np.mean(nonbonded_full_du_dls), np.std(nonbonded_full_du_dls)

        # keep the structure of grads the same as that of final_potentials so we can properly
        # form their vjps.
        grads = []
        for obs in du_dp_obs:
            grads.append(obs.avg_du_dp())

        return (bonded_mean, bonded_std), (nonbonded_mean, nonbonded_std), grads


# this class is serializable.
class AbsoluteFreeEnergy(BaseFreeEnergy):

    def __init__(self, mol, ff):
        """
        Compute the absolute free energy of a molecule via 4D decoupling.

        Parameters
        ----------
        mol: rdkit mol
            Ligand to be decoupled

        ff: ff.Forcefield
            Ligand forcefield

        """
        self.mol = mol
        self.ff = ff
        self.top = topology.BaseTopology(mol, ff)


    def prepare_host_edge(self, host_system, host_coords):
        """
        Prepares the host-edge system

        Parameters
        ----------
        host_system: openmm.System
            openmm System object to be deserialized

        host_coords: np.array
            Nx3 array of atomic coordinates

        Returns
        -------
        4 tuple
            bound_potentials, vjp_fns_and_handles, combined_masses, combined_coords

        """
        ligand_masses = [a.GetMass() for a in self.mol.GetAtoms()]
        ligand_coords = get_romol_conf(self.mol)

        host_bps, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)
        num_host_atoms = host_coords.shape[0]

        hgt = topology.HostGuestTopology(host_bps, self.top)

        ff_tuples = [
            [hgt.parameterize_harmonic_bond, (self.ff.hb_handle,)],
            [hgt.parameterize_harmonic_angle, (self.ff.ha_handle,)],
            [hgt.parameterize_periodic_torsion, (self.ff.pt_handle, self.ff.it_handle)],
            [hgt.parameterize_nonbonded, (self.ff.q_handle, self.ff.lj_handle)]
        ]

        final_potentials = []
        final_vjp_and_handles = []

        for fn, handles in ff_tuples:
            combined_params, vjp_fn, combined_potential = jax.vjp(fn, *[handle.params for handle in handles], has_aux=True)
            final_potentials.append(combined_potential.bind(combined_params))
            final_vjp_and_handles.append((vjp_fn, handles))

        combined_masses = np.concatenate([host_masses, ligand_masses])
        combined_coords = np.concatenate([host_coords, ligand_coords])

        return final_potentials, final_vjp_and_handles, combined_masses, combined_coords

    # this can be used for both the solvent leg and the complex leg
    def host_edge(self, lamb, host_system, host_coords, box, equil_steps=10000, prod_steps=100000):
        """
        Run equilibrium decoupling simulation at a given value of lambda in a host environment.

        Parameters
        ----------
        lamb: float [0, 1]
            0 is the fully interacting system, and 1 is the non-interacting system

        host_system: openmm.System
            OpenMM System object to be deserialized. The host can be simply a box of water, or a fully
            solvated protein

        host_coords: np.array of shape [..., 3]
            Host coordinates, in nanometers. It should be properly minimized and not have clashes
            with the ligand coordinates.

        box: np.array [3,3]
            Periodic boundary conditions, in nanometers.

        equil_steps: float
            Number of steps to run equilibration. Statistics are not gathered.

        prod_steps: float
            Number of steps to run production. Statistics are gathered.


        Returns
        -------
        float, float
            Returns a pair of average du_dl values for bonded and nonbonded terms.

        """
        final_potentials, final_vjp_and_handles, combined_masses, combined_coords = self.prepare_host_edge(host_system, host_coords)
    
        bonded_us, nonbonded_us, grads = self._simulate(
            lamb,
            box,
            combined_coords,
            np.zeros_like(combined_coords),
            final_potentials,
            self._get_integrator(combined_masses),
            equil_steps,
            prod_steps
        )

        return bonded_us, nonbonded_us, self._process_grads(grads, final_vjp_and_handles)


# this class is serializable.
class RelativeFreeEnergy(BaseFreeEnergy):

    def __init__(self, mol_a, mol_b, core, ff):
        self.mol_a = mol_a
        self.mol_b = mol_b
        self.core = core
        self.ff = ff
        self.top = topology.SingleTopology(mol_a, mol_b, core, ff)

    def _get_integrator(self, combined_masses):
        """
        Get a integrator. The resulting impl must be bound to a python handle
        whose lifetime is concurrent with that of the context.
        """
        seed = np.random.randint(np.iinfo(np.int32).max)

        return LangevinIntegrator(
            300.0,
            1.5e-3,
            1.0,
            combined_masses,
            seed
        )

    def prepare_vacuum_edge(self):
        """
        Prepares the vacuum system.

        Returns
        -------
        4 tuple
            bound_potentials, vjp_fns_and_handles, combined_masses, combined_coords

        """
        final_potentials = []
        final_vjp_and_handles = []

        ligand_masses_a = [a.GetMass() for a in self.mol_a.GetAtoms()]
        ligand_masses_b = [b.GetMass() for b in self.mol_b.GetAtoms()]

        ligand_coords_a = get_romol_conf(self.mol_a)
        ligand_coords_b = get_romol_conf(self.mol_b)

        ff_tuples = [
            [self.top.parameterize_harmonic_bond, (self.ff.hb_handle,)],
            [self.top.parameterize_harmonic_angle, (self.ff.ha_handle,)],
            [self.top.parameterize_periodic_torsion, (self.ff.pt_handle, self.ff.it_handle)],
            [self.top.parameterize_nonbonded, (self.ff.q_handle, self.ff.lj_handle)]
        ]

        final_potentials = []
        final_vjp_and_handles = []

        for fn, handles in ff_tuples:
            combined_params, vjp_fn, combined_potential = jax.vjp(fn, *[handle.params for handle in handles], has_aux=True)
            final_potentials.append(combined_potential.bind(combined_params))
            final_vjp_and_handles.append((vjp_fn, handles))

        combined_masses = np.mean(self.top.interpolate_params(ligand_masses_a, ligand_masses_b), axis=0)
        combined_coords = np.mean(self.top.interpolate_params(ligand_coords_a, ligand_coords_b), axis=0)

        return final_potentials, final_vjp_and_handles, combined_masses, combined_coords


    def vacuum_edge(self, lamb, equil_steps=10000, prod_steps=100000):
        """
        Run a vacuum decoupling simulation at a given value of lambda.

        Parameters
        ----------
        lamb: float [0, 1]
            0 is the fully interacting system, and 1 is the non-interacting system

        equil_steps: float
            Number of steps to run equilibration. Statistics are not gathered.

        prod_steps: float
            Number of steps to run production. Statistics are gathered.

        Returns
        -------
        float, float
            Returns a pair of average du_dl values for bonded and nonbonded terms.

        """
        final_potentials, final_vjp_and_handles, combined_masses, combined_coords = self.prepare_vacuum_edge()

        box = np.eye(3) * 100.0

        bonded_us, nonbonded_us, grads = self._simulate(
            lamb,
            box,
            combined_coords,
            np.zeros_like(combined_coords),
            final_potentials,
            self._get_integrator(combined_masses),
            equil_steps,
            prod_steps
        )

        return bonded_us, nonbonded_us, self._process_grads(grads, final_vjp_and_handles)

    def prepare_host_edge(self, host_system, host_coords):
        """
        Prepares the host-edge system

        Parameters
        ----------
        host_system: openmm.System
            openmm System object to be deserialized

        host_coords: np.array
            Nx3 array of atomic coordinates

        Returns
        -------
        4 tuple
            bound_potentials, vjp_fns_and_handles, combined_masses, combined_coords

        """

        ligand_masses_a = [a.GetMass() for a in self.mol_a.GetAtoms()]
        ligand_masses_b = [b.GetMass() for b in self.mol_b.GetAtoms()]

        # extract the 0th conformer
        ligand_coords_a = get_romol_conf(self.mol_a)
        ligand_coords_b = get_romol_conf(self.mol_b)

        host_bps, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)
        num_host_atoms = host_coords.shape[0]

        hgt = topology.HostGuestTopology(host_bps, self.top)

        ff_tuples = [
            [hgt.parameterize_harmonic_bond, (self.ff.hb_handle,)],
            [hgt.parameterize_harmonic_angle, (self.ff.ha_handle,)],
            [hgt.parameterize_periodic_torsion, (self.ff.pt_handle, self.ff.it_handle)],
            [hgt.parameterize_nonbonded, (self.ff.q_handle, self.ff.lj_handle)]
        ]

        final_potentials = []
        final_vjp_and_handles = []

        for fn, handles in ff_tuples:
            combined_params, vjp_fn, combined_potential = jax.vjp(fn, *[handle.params for handle in handles], has_aux=True)
            final_potentials.append(combined_potential.bind(combined_params))
            final_vjp_and_handles.append((vjp_fn, handles))

        combined_masses = np.concatenate([host_masses, np.mean(self.top.interpolate_params(ligand_masses_a, ligand_masses_b), axis=0)])
        combined_coords = np.concatenate([host_coords, np.mean(self.top.interpolate_params(ligand_coords_a, ligand_coords_b), axis=0)])

        return final_potentials, final_vjp_and_handles, combined_masses, combined_coords



    def host_edge(self, lamb, host_system, host_coords, box, equil_steps=10000, prod_steps=100000):
        """
        Run equilibrium decoupling simulation at a given value of lambda in a host environment.

        Parameters
        ----------
        lamb: float [0, 1]
            0 is the fully interacting system, and 1 is the non-interacting system

        host_system: openmm.System
            OpenMM System object to be deserialized. The host can be simply a box of water, or a fully
            solvated protein

        host_coords: np.array of shape [..., 3]
            Host coordinates, in nanometers. It should be properly minimized and not have clashes
            with the ligand coordinates.

        box: np.array [3,3]
            Periodic boundary conditions, in nanometers.

        equil_steps: float
            Number of steps to run equilibration. Statistics are not gathered.

        prod_steps: float
            Number of steps to run production. Statistics are gathered.

        Returns
        -------
        float, float
            Returns a pair of average du_dl values for bonded and nonbonded terms.

        """

        final_potentials, final_vjp_and_handles, combined_masses, combined_coords = self.prepare_host_edge(
            host_system, host_coords
        )

        # (ytz): us is short form for mean and std dev.
        bonded_us, nonbonded_us, grads = self._simulate(
            lamb,
            box,
            combined_coords,
            np.zeros_like(combined_coords),
            final_potentials,
            self._get_integrator(combined_masses),
            equil_steps,
            prod_steps
        )


        return bonded_us, nonbonded_us, self._process_grads(grads, final_vjp_and_handles)
