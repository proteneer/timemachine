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
import os

# this class is serializable.
class RelativeFreeEnergy():

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

    # this will be eventually gRPC'd out to a worker
    @staticmethod
    def _simulate(lamb, box, x0, v0, final_potentials, integrator, equil_steps, prod_steps):
        u_impls = []
        bonded_impls = []
        nonbonded_impls = []

        for bps in final_potentials:
            for bp in bps:
                impl = bp.bound_impl(np.float32)
                if isinstance(bp, potentials.InterpolatedPotential):
                    nonbonded_impls.append(impl)
                elif isinstance(bp, potentials.LambdaPotential):
                    bonded_impls.append(impl)
                u_impls.append(impl)

        intg_impl = integrator.impl()
        # context components: positions, velocities, box, integrator, energy fxns
        ctxt = custom_ops.Context(
            x0,
            v0,
            box,
            intg_impl,
            u_impls
        )

        # equilibration
        for step in range(equil_steps):
            ctxt.step(lamb)

        bonded_du_dl_obs = custom_ops.AvgPartialUPartialLambda(bonded_impls, 5)
        nonbonded_du_dl_obs = custom_ops.AvgPartialUPartialLambda(nonbonded_impls, 5)

        # add observable
        ctxt.add_observable(bonded_du_dl_obs)
        ctxt.add_observable(nonbonded_du_dl_obs)

        # du_dps = []
        # for ui in u_impls:
        #     du_dp_obs = custom_ops.AvgPartialUPartialParam(ui, 5)
        #     ctxt.add_observable(du_dp_obs)
        #     du_dps.append(du_dp_obs)

        for _ in range(prod_steps):
            ctxt.step(lamb)

        return bonded_du_dl_obs.avg_du_dl(), nonbonded_du_dl_obs.avg_du_dl()        

    def vacuum_edge(self, lamb, equil_steps=10000, prod_steps=100000):

        final_potentials = []
        final_vjp_and_handles = []

        ligand_masses_a = [a.GetMass() for a in self.mol_a.GetAtoms()]
        ligand_masses_b = [b.GetMass() for b in self.mol_b.GetAtoms()]

        ligand_coords_a = get_romol_conf(self.mol_a)
        ligand_coords_b = get_romol_conf(self.mol_b)

        bonded_tuples = [
            [self.top.parameterize_harmonic_bond, self.ff.hb_handle],
            [self.top.parameterize_harmonic_angle, self.ff.ha_handle],
            [self.top.parameterize_proper_torsion, self.ff.pt_handle],
            [self.top.parameterize_improper_torsion, self.ff.it_handle]
        ]

        # instantiate the vjps while parameterizing (forward pass)
        for fn, handle in bonded_tuples:
            (src_params, dst_params, uni_params), vjp_fn, (src_potential, dst_potential, uni_potential) = jax.vjp(fn, handle.params, has_aux=True)
            final_potentials.append([src_potential.bind(src_params), dst_potential.bind(dst_params), uni_potential.bind(uni_params)])
            final_vjp_and_handles.append((vjp_fn, handle))

        nb_params, vjp_fn, nb_potential = jax.vjp(self.top.parameterize_nonbonded, self.ff.q_handle.params, self.ff.lj_handle.params, has_aux=True)
        final_potentials.append([nb_potential.bind(nb_params)])
        final_vjp_and_handles.append([vjp_fn])

        combined_masses = np.mean(self.top.interpolate_params(ligand_masses_a, ligand_masses_b), axis=0)

        src_conf, dst_conf = self.top.interpolate_params(ligand_coords_a, ligand_coords_b)
        combined_coords = np.mean(self.top.interpolate_params(ligand_coords_a, ligand_coords_b), axis=0)

        box = np.eye(3) * 100.0

        return self._simulate(
            lamb,
            box,
            combined_coords,
            np.zeros_like(combined_coords),
            final_potentials,
            self._get_integrator(combined_masses),
            equil_steps,
            prod_steps
        )

    # this can be used for both the solvent leg and the complex leg
    def host_edge(self, lamb, host_system, host_coords, box, equil_steps=10000, prod_steps=100000):

        ligand_masses_a = [a.GetMass() for a in self.mol_a.GetAtoms()]
        ligand_masses_b = [b.GetMass() for b in self.mol_b.GetAtoms()]

        # extract the 0th conformer
        ligand_coords_a = get_romol_conf(self.mol_a)
        ligand_coords_b = get_romol_conf(self.mol_b)

        host_bps, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)
        num_host_atoms = host_coords.shape[0]

        final_potentials = []
        final_vjp_and_handles = []

        # keep the bonded terms in the host the same.
        # but we keep the nonbonded term for a subsequent modification
        for bp in host_bps:
            if isinstance(bp, potentials.Nonbonded):
                host_p = bp
            else:
                final_potentials.append([bp])
                final_vjp_and_handles.append(None)

        # gdt = topology.SingleTopology(self.mol_a, self.mol_b, core, ff)
        hgt = topology.HostGuestTopology(host_p, self.top)

        # setup the parameter handlers for the ligand
        bonded_tuples = [
            [hgt.parameterize_harmonic_bond, self.ff.hb_handle],
            [hgt.parameterize_harmonic_angle, self.ff.ha_handle],
            [hgt.parameterize_proper_torsion, self.ff.pt_handle],
            [hgt.parameterize_improper_torsion, self.ff.it_handle]
        ]

        # instantiate the vjps while parameterizing (forward pass)
        for fn, handle in bonded_tuples:
            (src_params, dst_params, uni_params), vjp_fn, (src_potential, dst_potential, uni_potential) = jax.vjp(fn, handle.params, has_aux=True)
            final_potentials.append([src_potential.bind(src_params), dst_potential.bind(dst_params), uni_potential.bind(uni_params)])
            final_vjp_and_handles.append((vjp_fn, handle))

        nb_params, vjp_fn, nb_potential = jax.vjp(hgt.parameterize_nonbonded, self.ff.q_handle.params, self.ff.lj_handle.params, has_aux=True)
        final_potentials.append([nb_potential.bind(nb_params)])
        final_vjp_and_handles.append([vjp_fn])

        combined_masses = np.concatenate([host_masses, np.mean(self.top.interpolate_params(ligand_masses_a, ligand_masses_b), axis=0)])

        # intg = self._get_integrator(combined_masses)

        src_conf, dst_conf = self.top.interpolate_params(ligand_coords_a, ligand_coords_b)
        combined_coords = np.concatenate([host_coords, np.mean(self.top.interpolate_params(ligand_coords_a, ligand_coords_b), axis=0)])

        return self._simulate(
            lamb,
            box,
            combined_coords,
            np.zeros_like(combined_coords),
            final_potentials,
            self._get_integrator(combined_masses),
            equil_steps,
            prod_steps
        )

        # x0 = combined_coords
        # v0 = np.zeros_like(x0)

        # u_impls = []
        # bonded_impls = []
        # nonbonded_impls = []

        # for bps in final_potentials:
        #     for bp in bps:
        #         impl = bp.bound_impl(np.float32)
        #         if isinstance(bp, potentials.InterpolatedPotential):
        #             nonbonded_impls.append(impl)
        #         elif isinstance(bp, potentials.LambdaPotential):
        #             bonded_impls.append(impl)
        #         u_impls.append(impl)

        # # context components: positions, velocities, box, integrator, energy fxns
        # ctxt = custom_ops.Context(
        #     x0,
        #     v0,
        #     box,
        #     intg,
        #     u_impls
        # )

        # for step in range(equil_steps):
        #     ctxt.step(lamb)

        # bonded_du_dl_obs = custom_ops.AvgPartialUPartialLambda(bonded_impls, 5)
        # nonbonded_du_dl_obs = custom_ops.AvgPartialUPartialLambda(nonbonded_impls, 5)

        # ctxt.add_observable(bonded_du_dl_obs)
        # ctxt.add_observable(nonbonded_du_dl_obs)

        # du_dps = []
        # for ui in u_impls:
        #     du_dp_obs = custom_ops.AvgPartialUPartialParam(ui, 5)
        #     ctxt.add_observable(du_dp_obs)
        #     du_dps.append(du_dp_obs)

        # for _ in range(prod_steps):
        #     ctxt.step(lamb)

        # print("host", bonded_du_dl_obs.avg_du_dl(), nonbonded_du_dl_obs.avg_du_dl())

        # return bonded_du_dl_obs.avg_du_dl(), nonbonded_du_dl_obs.avg_du_dl()
        # assert 0