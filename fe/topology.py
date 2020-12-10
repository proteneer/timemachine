import numpy as np
import jax.numpy as jnp

from timemachine.lib import potentials
from ff.handlers import nonbonded, bonded

_SCALE_12 = 1.0
_SCALE_13 = 1.0
_SCALE_14 = 0.5
_BETA = 2.0
_CUTOFF = 1.2

class HostGuestTopology():

    def __init__(self, guest_topology, host_p, num_host_atoms):
        """
        Utility tool for combining host with a guest, in that order.

        Parameters
        ----------
        guest_topology:
            Guest's Topology {Base, Dual, Single}Topology.

        host_p:
            Nonbonded potential for the host.

        num_host_atoms:
            Number of atoms in the host.
    
        """
        self.guest_topology = guest_topology
        self.host_p = host_p
        self.num_host_atoms = num_host_atoms

    def parameterize_harmonic_bond(self, ff_params):
        params, potential = self.guest_topology.parameterize_harmonic_bond(ff_params)
        potential.set_bond_idxs(potential.get_bond_idxs() + self.num_host_atoms)
        return params, potential

    def parameterize_harmonic_angle(self, ff_params):
        params, potential = self.guest_topology.parameterize_harmonic_angle(ff_params)
        potential.set_angle_idxs(potential.get_angle_idxs() + self.num_host_atoms)
        return params, potential

    def parameterize_proper_torsion(self, ff_params):
        params, potential = self.guest_topology.parameterize_proper_torsion(ff_params)
        potential.set_torsion_idxs(potential.get_torsion_idxs() + self.num_host_atoms)
        return params, potential

    def parameterize_improper_torsion(self, ff_params):
        params, potential = self.guest_topology.parameterize_improper_torsion(ff_params)
        potential.set_torsion_idxs(potential.get_torsion_idxs() + self.num_host_atoms)
        return params, potential

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):
        # this needs to take care of the case when there's parameter interpolation.
        num_guest_atoms = self.guest_topology.get_num_atoms()
        guest_qlj, guest_p = self.guest_topology.parameterize_nonbonded(ff_q_params, ff_lj_params)
        
        # see if we're doing parameter interpolation
        assert guest_qlj.shape[1] == 3

        assert guest_p.get_beta() == self.host_p.get_beta()
        assert guest_p.get_cutoff() == self.host_p.get_cutoff()

        if guest_qlj.shape[0] == num_guest_atoms:
            # no parameter interpolation
            hg_nb_params = jnp.concatenate([self.host_p.params, guest_qlj])
        elif guest_qlj.shape[0] == num_guest_atoms*2:
            # with parameter interpolation
            hg_nb_params_src = jnp.concatenate([self.host_p.params, guest_qlj[:num_guest_atoms]])
            hg_nb_params_dst = jnp.concatenate([self.host_p.params, guest_qlj[num_guest_atoms:]])
            hg_nb_params = jnp.concatenate([hg_nb_params_src, hg_nb_params_dst])
        else:
            # you dun' goofed and consequences will never be the same
            assert 0

        hg_exclusion_idxs = np.concatenate([self.host_p.get_exclusion_idxs(), guest_p.get_exclusion_idxs() + self.num_host_atoms])
        hg_scale_factors = np.concatenate([self.host_p.get_scale_factors(), guest_p.get_scale_factors()])
        hg_lambda_offset_idxs = np.concatenate([self.host_p.get_lambda_offset_idxs(), guest_p.get_lambda_offset_idxs()])
        hg_lambda_plane_idxs = np.concatenate([self.host_p.get_lambda_plane_idxs(), guest_p.get_lambda_plane_idxs()])

        return hg_nb_params, potentials.Nonbonded(
            hg_exclusion_idxs,
            hg_scale_factors,
            hg_lambda_plane_idxs,
            hg_lambda_offset_idxs,
            guest_p.get_beta(),
            guest_p.get_cutoff()
        )


class BaseTopology():

    def __init__(self, mol, forcefield):
        """
        Utility for working with a single ligand.

        Parameter
        ---------
        mol: ROMol
            Ligand to be parameterized

        forcefield: ff.Forcefield
            A convenience wrapper for forcefield lists.

        """
        self.mol = mol
        self.ff = forcefield

    def get_num_atoms(self):
        return self.mol.GetNumAtoms()

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):
        q_params = self.ff.q_handle.partial_parameterize(ff_q_params, self.mol)
        lj_params = self.ff.lj_handle.partial_parameterize(ff_lj_params, self.mol)

        exclusion_idxs, scale_factors = nonbonded.generate_exclusion_idxs(
            self.mol,
            scale12=_SCALE_12,
            scale13=_SCALE_13,
            scale14=_SCALE_14
        )

        scale_factors = np.stack([scale_factors, scale_factors], axis=1)

        N = len(q_params)

        lambda_plane_idxs = np.zeros(N, dtype=np.int32)
        lambda_offset_idxs = np.ones(N, dtype=np.int32)

        beta = _BETA
        cutoff = _CUTOFF # solve for this analytically later

        nb = potentials.Nonbonded(
            exclusion_idxs,
            scale_factors,
            lambda_plane_idxs,
            lambda_offset_idxs,
            beta,
            cutoff
        ) 

        params = jnp.concatenate([
            jnp.reshape(q_params, (-1, 1)),
            jnp.reshape(lj_params, (-1, 2))
        ], axis=1)

        return params, nb

    def parameterize_harmonic_bond(self, ff_params):
        params, idxs = self.ff.hb_handle.partial_parameterize(ff_params, self.mol)
        return params, potentials.HarmonicBond(idxs)


    def parameterize_harmonic_angle(self, ff_params):
        params, idxs = self.ff.ha_handle.partial_parameterize(ff_params, self.mol)
        return params, potentials.HarmonicAngle(idxs)


    def parameterize_proper_torsion(self, ff_params):
        params, idxs = self.ff.pt_handle.partial_parameterize(ff_params, self.mol)
        return params, potentials.PeriodicTorsion(idxs)


    def parameterize_improper_torsion(self, ff_params):
        params, idxs = self.ff.it_handle.partial_parameterize(ff_params, self.mol)
        return params, potentials.PeriodicTorsion(idxs)



class DualTopology():

    def __init__(self, mol_a, mol_b, forcefield):
        """
        Utility for working with two ligands via dual topology. Both copies of the ligand
        will be present after merging.

        Parameter
        ---------
        mol_a: ROMol
            First ligand to be parameterized

        mol_b: ROMol
            Second ligand to be parameterized

        forcefield: ff.Forcefield
            A convenience wrapper for forcefield lists.

        """
        self.mol_a = mol_a
        self.mol_b = mol_b
        self.ff = forcefield

    def get_num_atoms(self):
        return self.mol_a.GetNumAtoms() + self.mol_b.GetNumAtoms()

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):
        q_params_a = self.ff.q_handle.partial_parameterize(ff_q_params, self.mol_a)
        q_params_b = self.ff.q_handle.partial_parameterize(ff_q_params, self.mol_b) # HARD TYPO
        lj_params_a = self.ff.lj_handle.partial_parameterize(ff_lj_params, self.mol_a)
        lj_params_b = self.ff.lj_handle.partial_parameterize(ff_lj_params, self.mol_b)

        q_params = jnp.concatenate([q_params_a, q_params_b])
        lj_params = jnp.concatenate([lj_params_a, lj_params_b])

        exclusion_idxs_a, scale_factors_a = nonbonded.generate_exclusion_idxs(
            self.mol_a,
            scale12=_SCALE_12,
            scale13=_SCALE_13,
            scale14=_SCALE_14
        )

        exclusion_idxs_b, scale_factors_b = nonbonded.generate_exclusion_idxs(
            self.mol_b,
            scale12=_SCALE_12,
            scale13=_SCALE_13,
            scale14=_SCALE_14
        )

        mutual_exclusions = []
        mutual_scale_factors = []

        NA = self.mol_a.GetNumAtoms()
        NB = self.mol_b.GetNumAtoms()

        for i in range(NA):
            for j in range(NB):
                mutual_exclusions.append([i, j + NA])
                mutual_scale_factors.append([1.0, 1.0])

        mutual_exclusions = np.array(mutual_exclusions)
        mutual_scale_factors = np.array(mutual_scale_factors)

        combined_exclusion_idxs = np.concatenate([
            exclusion_idxs_a,
            exclusion_idxs_b + NA,
            mutual_exclusions
        ]).astype(np.int32)

        combined_scale_factors = np.concatenate([
            np.stack([scale_factors_a, scale_factors_a], axis=1),
            np.stack([scale_factors_b, scale_factors_b], axis=1),
            mutual_scale_factors
        ]).astype(np.float64)

        combined_lambda_plane_idxs = np.zeros(NA+NB, dtype=np.int32)
        combined_lambda_offset_idxs = np.concatenate([
            np.ones(NA, dtype=np.int32),
            np.ones(NB, dtype=np.int32)
        ])

        beta = _BETA
        cutoff = _CUTOFF # solve for this analytically later

        nb = potentials.Nonbonded(
            combined_exclusion_idxs,
            combined_scale_factors,
            combined_lambda_plane_idxs,
            combined_lambda_offset_idxs,
            beta,
            cutoff
        ) 

        params = jnp.concatenate([
            jnp.reshape(q_params, (-1, 1)),
            jnp.reshape(lj_params, (-1, 2))
        ], axis=1)

        return params, nb

    def parameterize_harmonic_bond(self, ff_params):
        offset = self.mol_a.GetNumAtoms()
        params_a, idxs_a = self.ff.hb_handle.partial_parameterize(ff_params, self.mol_a)
        params_b, idxs_b = self.ff.hb_handle.partial_parameterize(ff_params, self.mol_b)
        params_c = jnp.concatenate([params_a, params_b])
        idxs_c = jnp.concatenate([idxs_a, idxs_b + offset])
        return params_c, potentials.HarmonicBond(idxs_c)


    def parameterize_harmonic_angle(self, ff_params):
        offset = self.mol_a.GetNumAtoms()
        params_a, idxs_a = self.ff.ha_handle.partial_parameterize(ff_params, self.mol_a)
        params_b, idxs_b = self.ff.ha_handle.partial_parameterize(ff_params, self.mol_b)
        params_c = jnp.concatenate([params_a, params_b])
        idxs_c = jnp.concatenate([idxs_a, idxs_b + offset])
        return params_c, potentials.HarmonicAngle(idxs_c)


    def parameterize_proper_torsion(self, ff_params):
        offset = self.mol_a.GetNumAtoms()
        params_a, idxs_a = self.ff.pt_handle.partial_parameterize(ff_params, self.mol_a)
        params_b, idxs_b = self.ff.pt_handle.partial_parameterize(ff_params, self.mol_b)
        params_c = jnp.concatenate([params_a, params_b])
        idxs_c = jnp.concatenate([idxs_a, idxs_b + offset])
        return params_c, potentials.PeriodicTorsion(idxs_c)

    def parameterize_improper_torsion(self, ff_params):
        offset = self.mol_a.GetNumAtoms()
        params_a, idxs_a = self.ff.it_handle.partial_parameterize(ff_params, self.mol_a)
        params_b, idxs_b = self.ff.it_handle.partial_parameterize(ff_params, self.mol_b)
        params_c = jnp.concatenate([params_a, params_b])
        idxs_c = jnp.concatenate([idxs_a, idxs_b + offset])
        return params_c, potentials.PeriodicTorsion(idxs_c)

