import numpy as np
import jax
import jax.numpy as jnp

from timemachine.lib import potentials
from ff.handlers import nonbonded, bonded

_SCALE_12 = 1.0
_SCALE_13 = 1.0
_SCALE_14 = 0.5
_BETA = 2.0
_CUTOFF = 1.2

_MAX_PERIOD = 6
_MAX_PHASE = 4

class HostGuestTopology():

    def __init__(self, host_p, guest_topology):
        """
        Utility tool for combining host with a guest, in that order.

        Parameters
        ----------
        host_p:
            Nonbonded potential for the host.

        guest_topology:
            Guest's Topology {Base, Dual, Single}Topology.

        """
        self.guest_topology = guest_topology
        self.host_p = host_p
        self.num_host_atoms = len(self.host_p.get_lambda_plane_idxs())

    def get_num_atoms(self):
        return self.num_host_atoms + self.guest_topology.get_num_atoms()

    def _parameterize_bonded_term(self, ff_params, handle_fn):
        params, potential = handle_fn(ff_params)
        if len(params) == 3:
            src_params, dst_params, uni_params = params
            src_potential, dst_potential, uni_potential = potential
            src_potential.set_idxs(src_potential.get_idxs() + self.num_host_atoms)
            dst_potential.set_idxs(dst_potential.get_idxs() + self.num_host_atoms)
            uni_potential.set_idxs(uni_potential.get_idxs() + self.num_host_atoms)
            src_potential.set_N(src_potential.get_N() + self.num_host_atoms)
            dst_potential.set_N(dst_potential.get_N() + self.num_host_atoms)
            uni_potential.set_N(uni_potential.get_N() + self.num_host_atoms)

            return [src_params, dst_params, uni_params], [src_potential, dst_potential, uni_potential]
        else:
            potential.set_idxs(potential.get_idxs() + self.num_host_atoms)
            return params, potential

    def parameterize_harmonic_bond(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.guest_topology.parameterize_harmonic_bond)

    def parameterize_harmonic_angle(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.guest_topology.parameterize_harmonic_angle)

    def parameterize_proper_torsion(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.guest_topology.parameterize_proper_torsion)

    def parameterize_improper_torsion(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.guest_topology.parameterize_improper_torsion)

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):
        # this needs to take care of the case when there's parameter interpolation.
        num_guest_atoms = self.guest_topology.get_num_atoms()
        guest_qlj, guest_p = self.guest_topology.parameterize_nonbonded(ff_q_params, ff_lj_params)
        
        # see if we're doing parameter interpolation
        assert guest_qlj.shape[1] == 3

        assert guest_p.get_beta() == self.host_p.get_beta()
        assert guest_p.get_cutoff() == self.host_p.get_cutoff()

        hg_exclusion_idxs = np.concatenate([self.host_p.get_exclusion_idxs(), guest_p.get_exclusion_idxs() + self.num_host_atoms])
        hg_scale_factors = np.concatenate([self.host_p.get_scale_factors(), guest_p.get_scale_factors()])
        hg_lambda_offset_idxs = np.concatenate([self.host_p.get_lambda_offset_idxs(), guest_p.get_lambda_offset_idxs()])
        hg_lambda_plane_idxs = np.concatenate([self.host_p.get_lambda_plane_idxs(), guest_p.get_lambda_plane_idxs()])

        if guest_qlj.shape[0] == num_guest_atoms:
            # no parameter interpolation
            hg_nb_params = jnp.concatenate([self.host_p.params, guest_qlj])

            return hg_nb_params, potentials.Nonbonded(
                hg_exclusion_idxs,
                hg_scale_factors,
                hg_lambda_plane_idxs,
                hg_lambda_offset_idxs,
                guest_p.get_beta(),
                guest_p.get_cutoff()
            )

        elif guest_qlj.shape[0] == num_guest_atoms*2:
            # with parameter interpolation
            hg_nb_params_src = jnp.concatenate([self.host_p.params, guest_qlj[:num_guest_atoms]])
            hg_nb_params_dst = jnp.concatenate([self.host_p.params, guest_qlj[num_guest_atoms:]])
            hg_nb_params = jnp.concatenate([hg_nb_params_src, hg_nb_params_dst])

            nb = potentials.Nonbonded(
                hg_exclusion_idxs,
                hg_scale_factors,
                hg_lambda_plane_idxs,
                hg_lambda_offset_idxs,
                guest_p.get_beta(),
                guest_p.get_cutoff()
            )

            return hg_nb_params, potentials.InterpolatedPotential(nb, self.get_num_atoms(), hg_nb_params.size)

        else:
            # you dun' goofed and consequences will never be the same
            assert 0


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

    def _parameterize_bonded_term(self, ff_params, bonded_handle, potential):
        offset = self.mol_a.GetNumAtoms()
        params_a, idxs_a = bonded_handle.partial_parameterize(ff_params, self.mol_a)
        params_b, idxs_b = bonded_handle.partial_parameterize(ff_params, self.mol_b)
        params_c = jnp.concatenate([params_a, params_b])
        idxs_c = jnp.concatenate([idxs_a, idxs_b + offset])
        return params_c, potential(idxs_c)

    def parameterize_harmonic_bond(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.hb_handle, potentials.HarmonicBond)


    def parameterize_harmonic_angle(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.ha_handle, potentials.HarmonicAngle)


    def parameterize_proper_torsion(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.pt_handle, potentials.PeriodicTorsion)


    def parameterize_improper_torsion(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.it_handle, potentials.PeriodicTorsion)


class SingleTopology():

    def __init__(self, mol_a, mol_b, core, ff):
        """
        SingleTopology combines two molecules through a common core.
        """
        self.mol_a = mol_a
        self.mol_b = mol_b
        self.ff = ff
        self.core = core

        assert core.shape[1] == 2

        # map into idxs in the combined molecule
        self.a_to_c = np.arange(mol_a.GetNumAtoms(), dtype=np.int32) # identity
        self.b_to_c = np.zeros(mol_b.GetNumAtoms(), dtype=np.int32) - 1

        self.NC = mol_a.GetNumAtoms() + mol_b.GetNumAtoms() - len(core)

        # mark membership:
        # 0: Core
        # 1: R_A (default)
        # 2: R_B
        self.c_flags = np.ones(self.get_num_atoms(), dtype=np.int32)

        for a, b in core:
            self.c_flags[a] = 0
            self.b_to_c[b] = a

        iota = self.mol_a.GetNumAtoms()
        for b_idx, c_idx in enumerate(self.b_to_c):
            if c_idx == -1:
                self.b_to_c[b_idx] = iota
                self.c_flags[iota] = 2
                iota += 1

        # test for uniqueness
        # (ytz): fix me for np arrays

        # print(core[:, 0])
        # print(tuple(core[:, 0]))

        # assert len(set(tuple(core[:, 0]))) == len(core[:, 0])
        # assert len(set(tuple(core[:, 1]))) == len(core[:, 1])

    def get_num_atoms(self):
        return self.NC

    def interpolate_params(self, params_a, params_b):
        """
        Interpolate two sets of per-particle parameters.

        This can be used to interpolate nonbonded parameters,
        coordinates, etc.

        Parameters
        ----------
        params_a: np.ndarray, shape [N_A, ...]
            Parameters for the mol_a

        params_b: np.ndarray, shape [N_B, ...]
            Parameters for the mol_b

        Returns
        -------
        tuple: (src, dst)
            Two np.ndarrays each of shape [N_C, ...]

        """

        src_params = [None]*self.get_num_atoms()
        dst_params = [None]*self.get_num_atoms()

        for a_idx, c_idx in enumerate(self.a_to_c):
            src_params[c_idx] = params_a[a_idx]
            dst_params[c_idx] = params_a[a_idx]

        for b_idx, c_idx in enumerate(self.b_to_c):
            dst_params[c_idx] = params_b[b_idx]
            if src_params[c_idx] is None:
                src_params[c_idx] = params_b[b_idx]

        return jnp.array(src_params), jnp.array(dst_params)


    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):
        q_params_a = self.ff.q_handle.partial_parameterize(ff_q_params, self.mol_a)
        q_params_b = self.ff.q_handle.partial_parameterize(ff_q_params, self.mol_b) # HARD TYPO
        lj_params_a = self.ff.lj_handle.partial_parameterize(ff_lj_params, self.mol_a)
        lj_params_b = self.ff.lj_handle.partial_parameterize(ff_lj_params, self.mol_b)

        qlj_params_a = jnp.concatenate([
            jnp.reshape(q_params_a, (-1, 1)),
            jnp.reshape(lj_params_a, (-1, 2))
        ], axis=1)
        qlj_params_b = jnp.concatenate([
            jnp.reshape(q_params_b, (-1, 1)),
            jnp.reshape(lj_params_b, (-1, 2))
        ], axis=1)

        qlj_params_src, qlj_params_dst = self.interpolate_params(qlj_params_a, qlj_params_b)
        qlj_params = jnp.concatenate([qlj_params_src, qlj_params_dst])

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

        # use the same scale factors of LJ & charges for now
        scale_factors_a = np.stack([scale_factors_a, scale_factors_a], axis=1)
        scale_factors_b = np.stack([scale_factors_b, scale_factors_b], axis=1)


        combined_exclusion_dict = dict()

        for ij, scale in zip(exclusion_idxs_a, scale_factors_a):
            ij = tuple(sorted(self.a_to_c[ij]))
            if ij in combined_exclusion_dict:
                np.testing.assert_array_equal(combined_exclusion_dict[ij], scale)
            else:
                combined_exclusion_dict[ij] = scale

        for ij, scale in zip(exclusion_idxs_b, scale_factors_b):
            ij = tuple(sorted(self.b_to_c[ij]))
            if ij in combined_exclusion_dict:
                np.testing.assert_array_equal(combined_exclusion_dict[ij], scale)
            else:
                combined_exclusion_dict[ij] = scale

        combined_exclusion_idxs = []
        combined_scale_factors = []

        for e, s in combined_exclusion_dict.items():
            combined_exclusion_idxs.append(e)
            combined_scale_factors.append(s)

        combined_exclusion_idxs = np.array(combined_exclusion_idxs)
        combined_scale_factors = np.array(combined_scale_factors)

        # (ytz): we don't need exclusions between R_A and R_B will never see each other
        # under this decoupling scheme. They will always be at cutoff apart from each other.

        # plane_idxs: RA = Core = 0, RB = -1
        # offset_idxs: Core = 0, RA = RB = +1 
        combined_lambda_plane_idxs = np.zeros(self.get_num_atoms(), dtype=np.int32)
        combined_lambda_offset_idxs = np.zeros(self.get_num_atoms(), dtype=np.int32)

        for atom, group in enumerate(self.c_flags):
            if group == 0:
                # core atom
                combined_lambda_plane_idxs[atom] = 0
                combined_lambda_offset_idxs[atom] = 0
            elif group == 1:
                combined_lambda_plane_idxs[atom] = 0
                combined_lambda_offset_idxs[atom] = 1
            elif group == 2:
                combined_lambda_plane_idxs[atom] = -1
                combined_lambda_offset_idxs[atom] = 1
            else:
                assert 0

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

        return qlj_params, potentials.InterpolatedPotential(nb, self.get_num_atoms(), qlj_params.size)

    def _parameterize_bonded_term(self, ff_params, bonded_handle, potential):
        params_a, idxs_a = bonded_handle.partial_parameterize(ff_params, self.mol_a)
        params_b, idxs_b = bonded_handle.partial_parameterize(ff_params, self.mol_b)

        core_params_a = []
        core_params_b = []
        unique_params_r = [] # always on

        core_idxs_a = []
        core_idxs_b = []
        unique_idxs_r = [] # always on

        for p, old_atoms in zip(params_a, idxs_a):
            new_atoms = self.a_to_c[old_atoms]
            if np.all(self.c_flags[new_atoms] == 0):
                core_params_a.append(p)
                core_idxs_a.append(new_atoms)
            else:
                unique_params_r.append(p)
                unique_idxs_r.append(new_atoms)

        for p, old_atoms in zip(params_b, idxs_b):
            new_atoms = self.b_to_c[old_atoms]
            if np.all(self.c_flags[new_atoms] == 0):
                core_params_b.append(p)
                core_idxs_b.append(new_atoms)
            else:
                unique_params_r.append(p)
                unique_idxs_r.append(new_atoms)

        core_params_a = jnp.array(core_params_a)
        core_params_b = jnp.array(core_params_b)
        unique_params_r = jnp.array(unique_params_r) # always on

        core_idxs_a = np.array(core_idxs_a, dtype=np.int32)
        core_idxs_b = np.array(core_idxs_b, dtype=np.int32)

        # for i, p in zip(core_idxs_a, core_params_a):
            # print(i, repr(p))

        # for i, p in zip(core_idxs_b, core_params_b):
            # print(i, p)
        # print(core_idxs_a)
        # print(core_idxs_b)

        # assert 0

        unique_idxs_r = np.array(unique_idxs_r, dtype=np.int32) # always on

        u_fn_a = potentials.LambdaPotential(potential(core_idxs_a), self.get_num_atoms(), core_params_a.size, -1.0, 1.0)
        u_fn_b = potentials.LambdaPotential(potential(core_idxs_b), self.get_num_atoms(), core_params_b.size,  1.0, 0.0)
        u_fn_r = potentials.LambdaPotential(potential(unique_idxs_r), self.get_num_atoms(), unique_params_r.size,  0.0, 1.0)

        return [core_params_a, core_params_b, unique_params_r], [u_fn_a, u_fn_b, u_fn_r]

    def parameterize_harmonic_bond(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.hb_handle, potentials.HarmonicBond)


    def parameterize_harmonic_angle(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.ha_handle, potentials.HarmonicAngle)


    def parameterize_proper_torsion(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.pt_handle, potentials.PeriodicTorsion)


    def parameterize_improper_torsion(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.it_handle, potentials.PeriodicTorsion)

