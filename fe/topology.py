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


class AtomMappingError(Exception):
    pass

class UnsupportedPotential(Exception):
    pass

class HostGuestTopology():

    def __init__(self, 
        host_potentials,
        guest_topology):
        """
        Utility tool for combining host with a guest, in that order. host_potentials must be comprised
        exclusively of supported potentials (currently: bonds, angles, torsions, nonbonded).

        Parameters
        ----------
        host_potentials:
            Bound potentials for the host.

        guest_topology:
            Guest's Topology {Base, Dual, Single}Topology.

        """
        self.guest_topology = guest_topology

        self.host_nonbonded = None
        self.host_harmonic_bond = None
        self.host_harmonic_angle = None
        self.host_periodic_torsion = None

        # (ytz): extra assertions inside are to ensure we don't have duplicate terms
        for bp in host_potentials:
            if isinstance(bp, potentials.HarmonicBond):
                assert self.host_harmonic_bond is None
                self.host_harmonic_bond = bp
            elif isinstance(bp, potentials.HarmonicAngle):
                assert self.host_harmonic_angle is None
                self.host_harmonic_angle = bp
            elif isinstance(bp, potentials.PeriodicTorsion):
                assert self.host_periodic_torsion is None
                self.host_periodic_torsion = bp
            elif isinstance(bp, potentials.Nonbonded):
                assert self.host_nonbonded is None
                self.host_nonbonded = bp
            else:
                raise UnsupportedPotential("Unsupported host potential")

        self.num_host_atoms = len(self.host_nonbonded.get_lambda_plane_idxs())

    def get_num_atoms(self):
        return self.num_host_atoms + self.guest_topology.get_num_atoms()

    # tbd: just merge the hamiltonians here
    def _parameterize_bonded_term(self, guest_params, guest_potential, host_potential):

        if guest_potential is None:
            raise UnsupportedPotential("Mismatch in guest_potential")

        # (ytz): corner case exists if the guest_potential is None
        if host_potential is not None:
            assert type(host_potential) == type(guest_potential)

        guest_idxs = guest_potential.get_idxs() + self.num_host_atoms

        guest_lambda_mult = guest_potential.get_lambda_mult()
        guest_lambda_offset = guest_potential.get_lambda_offset()

        if guest_lambda_mult is None:
            guest_lambda_mult = np.zeros(len(guest_params))
        if guest_lambda_offset is None:
            guest_lambda_offset = np.ones(len(guest_params))

        if host_potential is not None:
            # the host is always on.
            host_params = host_potential.params
            host_idxs = host_potential.get_idxs()
            host_lambda_mult = jnp.zeros(len(host_idxs), dtype=np.int32)
            host_lambda_offset = jnp.ones(len(host_idxs), dtype=np.int32)
        else:
            # (ytz): this extra jank is to work around jnp.concatenate not supporting empty lists.
            host_params = np.array([], dtype=guest_params.dtype).reshape((-1, guest_params.shape[1]))
            host_idxs = np.array([], dtype=guest_idxs.dtype).reshape((-1, guest_idxs.shape[1]))
            host_lambda_mult = []
            host_lambda_offset = []

        combined_params = jnp.concatenate([host_params, guest_params])
        combined_idxs = np.concatenate([host_idxs, guest_idxs])
        combined_lambda_mult = np.concatenate([host_lambda_mult, guest_lambda_mult]).astype(np.int32)
        combined_lambda_offset = np.concatenate([host_lambda_offset, guest_lambda_offset]).astype(np.int32)

        ctor = type(guest_potential)

        return combined_params, ctor(combined_idxs, combined_lambda_mult, combined_lambda_offset)

    def parameterize_harmonic_bond(self, ff_params):
        guest_params, guest_potential = self.guest_topology.parameterize_harmonic_bond(ff_params)
        return self._parameterize_bonded_term(guest_params, guest_potential, self.host_harmonic_bond)

    def parameterize_harmonic_angle(self, ff_params):
        guest_params, guest_potential = self.guest_topology.parameterize_harmonic_angle(ff_params)
        return self._parameterize_bonded_term(guest_params, guest_potential, self.host_harmonic_angle)

    def parameterize_periodic_torsion(self, proper_params, improper_params):
        guest_params, guest_potential = self.guest_topology.parameterize_periodic_torsion(proper_params, improper_params)
        return self._parameterize_bonded_term(guest_params, guest_potential, self.host_periodic_torsion)

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):
        num_guest_atoms = self.guest_topology.get_num_atoms()
        guest_qlj, guest_p = self.guest_topology.parameterize_nonbonded(ff_q_params, ff_lj_params)

        if isinstance(guest_p, potentials.InterpolatedPotential):
            assert guest_qlj.shape[0] == num_guest_atoms*2
            guest_p = guest_p.get_u_fn()
            is_interpolated = True
        else:
            assert guest_qlj.shape[0] == num_guest_atoms
            is_interpolated = False

        # see if we're doing parameter interpolation
        assert guest_qlj.shape[1] == 3
        assert guest_p.get_beta() == self.host_nonbonded.get_beta()
        assert guest_p.get_cutoff() == self.host_nonbonded.get_cutoff()

        hg_exclusion_idxs = np.concatenate([self.host_nonbonded.get_exclusion_idxs(), guest_p.get_exclusion_idxs() + self.num_host_atoms])
        hg_scale_factors = np.concatenate([self.host_nonbonded.get_scale_factors(), guest_p.get_scale_factors()])
        hg_lambda_offset_idxs = np.concatenate([self.host_nonbonded.get_lambda_offset_idxs(), guest_p.get_lambda_offset_idxs()])
        hg_lambda_plane_idxs = np.concatenate([self.host_nonbonded.get_lambda_plane_idxs(), guest_p.get_lambda_plane_idxs()])

        if is_interpolated:
            # with parameter interpolation
            hg_nb_params_src = jnp.concatenate([self.host_nonbonded.params, guest_qlj[:num_guest_atoms]])
            hg_nb_params_dst = jnp.concatenate([self.host_nonbonded.params, guest_qlj[num_guest_atoms:]])
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
            # no parameter interpolation
            hg_nb_params = jnp.concatenate([self.host_nonbonded.params, guest_qlj])

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

    def parameterize_periodic_torsion(self, proper_params, improper_params):
        """
        Parameterize all periodic torsions in the system.
        """
        proper_params, proper_potential = self.parameterize_proper_torsion(proper_params)
        improper_params, improper_potential = self.parameterize_improper_torsion(improper_params)
        combined_params = jnp.concatenate([proper_params, improper_params])
        combined_idxs = np.concatenate([proper_potential.get_idxs(), improper_potential.get_idxs()])
        combined_potential = potentials.PeriodicTorsion(combined_idxs)
        return combined_params, combined_potential


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
        q_params_b = self.ff.q_handle.partial_parameterize(ff_q_params, self.mol_b)
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
        idxs_c = np.concatenate([idxs_a, idxs_b + offset])
        return params_c, potential(idxs_c)

    def parameterize_harmonic_bond(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.hb_handle, potentials.HarmonicBond)


    def parameterize_harmonic_angle(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.ha_handle, potentials.HarmonicAngle)

    def parameterize_periodic_torsion(self, proper_params, improper_params):
        """
        Parameterize all periodic torsions in the system.
        """
        proper_params, proper_potential = self.parameterize_proper_torsion(proper_params)
        improper_params, improper_potential = self.parameterize_improper_torsion(improper_params)
        combined_params = jnp.concatenate([proper_params, improper_params])
        combined_idxs = np.concatenate([proper_potential.get_idxs(), improper_potential.get_idxs()])
        combined_potential = potentials.PeriodicTorsion(combined_idxs)
        return combined_params, combined_potential

    def parameterize_proper_torsion(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.pt_handle, potentials.PeriodicTorsion)

    def parameterize_improper_torsion(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.it_handle, potentials.PeriodicTorsion)


class SingleTopology():

    def __init__(self, mol_a, mol_b, core, ff):
        """
        SingleTopology combines two molecules through a common core. The combined mol has
        atom indices laid out such that mol_a is identically mapped to the combined mol indices.
        The atoms in the mol_b's R-group is then glued on to resulting molecule.

        Parameters
        ----------
        mol_a: ROMol
            First ligand

        mol_b: ROMol
            Second ligand

        core: np.array (C, 2)
            Atom mapping from mol_a to to mol_b

        ff: ff.Forcefield
            Forcefield to be used for parameterization.

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

        # test for uniqueness in core idxs for each mol
        assert len(set(tuple(core[:, 0]))) == len(core[:, 0])
        assert len(set(tuple(core[:, 1]))) == len(core[:, 1])

        self.assert_factorizability()

    def _identify_offending_core_indices(self):
        """Identifies atoms involved in violations of a factorizability assumption,
            but doesn't immediately raise an error.
            Later, could use this list to:
            * plot / debug
            * if in a "repair_mode", attempt to repair the mapping by removing offending atoms
            * otherwise, raise atom mapping error if any atoms were identified
        """

        # Test that R-groups can be properly factorized out in the proposed
        # mapping. The requirement is that R-groups must be branched from exactly
        # a single atom on the core.

        offending_core_indices = []

        # first convert to a dense graph
        N = self.get_num_atoms()
        dense_graph = np.zeros((N, N), dtype=np.int32)

        for bond in self.mol_a.GetBonds():
            i, j = self.a_to_c[bond.GetBeginAtomIdx()], self.a_to_c[bond.GetEndAtomIdx()]
            dense_graph[i, j] = 1
            dense_graph[j, i] = 1

        for bond in self.mol_b.GetBonds():
            i, j = self.b_to_c[bond.GetBeginAtomIdx()], self.b_to_c[bond.GetEndAtomIdx()]
            dense_graph[i, j] = 1
            dense_graph[j, i] = 1

            # sparsify to simplify and speed up traversal code
        sparse_graph = []
        for row in dense_graph:
            nbs = []
            for col_idx, col in enumerate(row):
                if col == 1:
                    nbs.append(col_idx)
            sparse_graph.append(nbs)

        def visit(i, visited):
            if i in visited:
                return
            else:
                visited.add(i)
                if self.c_flags[i] != 0:
                    for nb in sparse_graph[i]:
                        visit(nb, visited)
                else:
                    return

        for c_idx, group in enumerate(self.c_flags):
            # 0 core, 1 R_A, 2: R_B
            if group != 0:
                seen = set()
                visit(c_idx, seen)
                # (ytz): exactly one of seen should belong to core
                if np.sum(np.array([self.c_flags[x] for x in seen]) == 0) != 1:
                    offending_core_indices.append(c_idx)

        return offending_core_indices


    def assert_factorizability(self):
        """
        Number of atoms in the combined mol

        TODO: add a reference to Boresch paper describing the assumption being checked
        """
        offending_core_indices = self._identify_offending_core_indices()
        num_problems = len(offending_core_indices)
        if num_problems > 0:

            # TODO: revisit how to get atom pair indices -- this goes out of bounds
            # bad_pairs = [tuple(self.core[c_index]) for c_index in offending_core_indices]

            message = f"""Atom Mapping Error: the resulting map is non-factorizable!
            (The map contained  {num_problems} violations of the factorizability assumption.)
            """
            raise AtomMappingError(message)


    def get_num_atoms(self):
        return self.NC

    def interpolate_params(self, params_a, params_b):
        """
        Interpolate two sets of per-particle parameters.

        This can be used to interpolate masses, coordinates, etc.

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

    def interpolate_nonbonded_params(self, params_a, params_b):
        """
        Special interpolation method for nonbonded parameters. For R-group atoms,
        their charges and vdw eps parameters are scaled to zero. Vdw sigma
        remains unchanged. This method is needed in order to ensure that R-groups
        that branch from multiple distinct attachment points are fully non-interacting
        to allow for factorization of the partition function. In order words, this function
        implements essentially the non-softcore part of parameter interpolation.

        Parameters
        ----------
        params_a: np.ndarray, shape [N_A, 3]
            Nonbonded parameters for the mol_a

        params_b: np.ndarray, shape [N_B, 3]
            Nonbonded parameters for the mol_b

        Returns
        -------
        tuple: (src, dst)
            Two np.ndarrays each of shape [N_C, ...]

        """

        src_params = [None]*self.get_num_atoms()
        dst_params = [None]*self.get_num_atoms()

        # src -> dst is turning off the parameter
        for a_idx, c_idx in enumerate(self.a_to_c):
            params = params_a[a_idx]
            src_params[c_idx] = params
            if self.c_flags[c_idx] != 0:
                assert self.c_flags[c_idx] == 1
                dst_params[c_idx] = jnp.array([0, params[1], 0]) # q, sig, eps

        # b is initially decoupled
        for b_idx, c_idx in enumerate(self.b_to_c):
            params = params_b[b_idx]
            dst_params[c_idx] = params
            # this will already be processed when looping over a
            if self.c_flags[c_idx] == 0:
                assert src_params[c_idx] is not None
            else:
                assert self.c_flags[c_idx] == 2
                src_params[c_idx] = jnp.array([0, params[1], 0]) # q, sig, eps

        return jnp.array(src_params), jnp.array(dst_params)


    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):
        # Nonbonded potentials combine through parameter interpolation, not energy interpolation.
        # They may or may not operate through 4D decoupling depending on the atom mapping. If an atom is
        # unique, it is kept at full strength and not switched off.

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

        qlj_params_src, qlj_params_dst = self.interpolate_nonbonded_params(qlj_params_a, qlj_params_b)
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

        # (ytz): use the same scale factors of LJ & charges for now
        # this isn't quite correct as the LJ/Coluomb may be different in 
        # different forcefields.
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

    @staticmethod
    def _concatenate(arrs):
        non_empty = []
        for arr in arrs:
            if len(arr) != 0:
                non_empty.append(jnp.array(arr))
        return jnp.concatenate(non_empty)

    def _parameterize_bonded_term(self, ff_params, bonded_handle, potential):
        # Bonded terms are defined as follows:
        # If a bonded term is comprised exclusively of atoms in the core region, then
        # its energy its interpolated from the on-state to off-state.
        # Otherwise (i.e. it has one atom that is not in the core region), the bond term
        # is defined as unique, and is on at all times.
        # This means that the end state will contain dummy atoms that is not the true end-state,
        # but contains an analytical correction (through Boresch) that can be cancelled out.

        params_a, idxs_a = bonded_handle.partial_parameterize(ff_params, self.mol_a)
        params_b, idxs_b = bonded_handle.partial_parameterize(ff_params, self.mol_b)

        core_params_a = []
        core_params_b = []
        unique_params_r = []

        core_idxs_a = []
        core_idxs_b = []
        unique_idxs_r = []
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
        unique_params_r = jnp.array(unique_params_r)

        # number of parameters per term (2 for bonds, 2 for angles, 3 for torsions)
        P = params_a.shape[-1] # TODO: note P unused

        combined_params = self._concatenate([
            core_params_a,
            core_params_b,
            unique_params_r
        ])

        # number of atoms involved in the bonded term
        K = idxs_a.shape[-1]

        core_idxs_a = np.array(core_idxs_a, dtype=np.int32).reshape((-1, K))
        core_idxs_b = np.array(core_idxs_b, dtype=np.int32).reshape((-1, K))
        unique_idxs_r = np.array(unique_idxs_r, dtype=np.int32).reshape((-1, K)) # always on

        # TODO: assert `len(core_idxs_a) == len(core_idxs_b)` in a more fine-grained way

        combined_idxs = np.concatenate([core_idxs_a, core_idxs_b, unique_idxs_r])

        lamb_mult = np.array([-1]*len(core_idxs_a) + [1]*len(core_idxs_b) + [0]*len(unique_idxs_r), dtype=np.int32)
        lamb_offset = np.array([1]*len(core_idxs_a) + [0]*len(core_idxs_b) + [1]*len(unique_idxs_r), dtype=np.int32)

        u_fn = potential(combined_idxs, lamb_mult, lamb_offset)
        return combined_params, u_fn


    def parameterize_harmonic_bond(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.hb_handle, potentials.HarmonicBond)


    def parameterize_harmonic_angle(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.ha_handle, potentials.HarmonicAngle)


    def parameterize_proper_torsion(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.pt_handle, potentials.PeriodicTorsion)

    def parameterize_improper_torsion(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.it_handle, potentials.PeriodicTorsion)

    def parameterize_periodic_torsion(self, proper_params, improper_params):
        """
        Parameterize all periodic torsions in the system.
        """
        proper_params, proper_potential = self.parameterize_proper_torsion(proper_params)
        improper_params, improper_potential = self.parameterize_improper_torsion(improper_params)
        combined_params = jnp.concatenate([proper_params, improper_params])
        combined_idxs = np.concatenate([proper_potential.get_idxs(), improper_potential.get_idxs()])
        combined_lambda_mult = np.concatenate([proper_potential.get_lambda_mult(), improper_potential.get_lambda_mult()])
        combined_lambda_offset = np.concatenate([proper_potential.get_lambda_offset(), improper_potential.get_lambda_offset()])
        combined_potential = potentials.PeriodicTorsion(combined_idxs, combined_lambda_mult, combined_lambda_offset)
        return combined_params, combined_potential

