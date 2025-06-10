from dataclasses import replace
from functools import cached_property

import jax.numpy as jnp
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from openmm import app
from rdkit import Chem

from timemachine.constants import NBParamIdx
from timemachine.fe.single_topology import AlignedPotential, AtomMapFlags, SingleTopology
from timemachine.fe.system import GuestSystem, HostGuestSystem, HostSystem
from timemachine.ff import Forcefield
from timemachine.graph_utils import convert_to_nx

from .bond import CanonicalBond, CanonicalProper, mkbond, mkproper
from .interpolation import InterpolationFxn, InterpolationFxnName, Symmetric, get_interpolation_fxn
from .queries import get_aliphatic_ring_bonds, get_rotatable_bonds


def get_temperature_scale_interpolation_fxn(
    max_temperature_scale: float, interpolation: InterpolationFxnName
) -> InterpolationFxn:
    f = get_interpolation_fxn(interpolation, 1.0, max_temperature_scale)
    f = Symmetric(f)
    return f


class SingleTopologyREST(SingleTopology):
    r"""Extends :py:class`timemachine.fe.single_topology.SingleTopology` to enhance intermediate states with REST-like
    energy scaling of certain interactions.

    The enhanced interactions are:

    1. ligand proper torsions (i, j, k, l) where (j, k) is a rotatable bond
    2. ligand proper torsions (i, j, k, l) where (j, k) is an aliphatic ring bond
    3. ligand-ligand electrostatics and LJ
    4. ligand-environment electrostatics LJ

    Energy scaling is implemented by scaling interaction parameters by ``1 / temperature_scale``, where
    ``temperature_scale`` is interpolated using a function ``f`` with the following properties:

    - f(0) = f(1) = 1 (identity at end states)
    - f(0.5) = max_temperature_scale
    - f(x) = f(1 - x) (symmetric)
    """

    def __init__(
        self,
        mol_a: Chem.Mol,
        mol_b: Chem.Mol,
        core: NDArray,
        forcefield: Forcefield,
        max_temperature_scale: float,
        temperature_scale_interpolation: InterpolationFxnName = "exponential",
    ):
        """
        Parameters
        ----------
        mol_a: Chem.Mol
            First guest

        mol_b: Chem.Mol
            Second guest

        core: np.array (C, 2)
            Atom mapping from mol_a to mol_b

        forcefield: ff.Forcefield
            Forcefield to use for parameterization

        max_temperature_scale: float
            Maximum temperature scale factor

        temperature_scale_interpolation: str
            Interpolation function to use for temperature scaling. One of "linear", "quadratic", or "exponential"
        """
        super().__init__(mol_a, mol_b, core, forcefield)
        print("rest is enabled with temp:", max_temperature_scale)
        self._temperature_scale_interpolation_fxn: InterpolationFxn = get_temperature_scale_interpolation_fxn(
            max_temperature_scale, temperature_scale_interpolation
        )
        self._nxg_a = convert_to_nx(mol_a)
        self._nxg_b = convert_to_nx(mol_b)
        self._cycles_a = nx.cycle_basis(self._nxg_a)
        self._cycles_b = nx.cycle_basis(self._nxg_b)

    # expand REST region to include complete ring groups
    @staticmethod
    def expand_rest_region_in_mol(atom_idxs, cycles, mol):
        region = set()
        for atom_idx in atom_idxs:
            for cycle in cycles:
                if atom_idx in cycle:
                    for cycle_atom in cycle:
                        region.add(cycle_atom)

        # find terminal, 1-connected atoms that are not in the REST region, and add them.
        inner_rest_idxs = region.union(set(atom_idxs))
        outer_rest_idxs = set()
        for atom in mol.GetAtoms():
            nbs = atom.GetNeighbors()
            if len(nbs) == 1:
                nb = nbs[0].GetIdx()
                if nb in inner_rest_idxs:
                    outer_rest_idxs.add(atom.GetIdx())
            elif len(nbs) == 2:
                # special case to deal with 1-connected nitriles, hydroxyls
                nb_nb = None
                if nbs[0].GetIdx() in inner_rest_idxs:
                    nb_nb = nbs[1]
                elif nbs[1].GetIdx() in inner_rest_idxs:
                    nb_nb = nbs[0]

                if nb_nb is not None and nb_nb.GetDegree() == 1:
                    outer_rest_idxs.add(atom.GetIdx())
                    outer_rest_idxs.add(nb_nb.GetIdx())

        return inner_rest_idxs.union(outer_rest_idxs)

    def split_combined_idxs(self, combined_idxs):
        mol_a_idxs = []
        for idx in combined_idxs:
            if self.c_flags[idx] == AtomMapFlags.CORE or self.c_flags[idx] == AtomMapFlags.MOL_A:
                mol_a_idxs.append(self.c_to_a[idx])

        mol_b_idxs = []
        for idx in combined_idxs:
            if self.c_flags[idx] == AtomMapFlags.CORE or self.c_flags[idx] == AtomMapFlags.MOL_B:
                mol_b_idxs.append(self.c_to_b[idx])

        return mol_a_idxs, mol_b_idxs

    @cached_property
    def base_rest_region_atom_idxs(self) -> set[int]:
        """Returns the set of indices of atoms in the combined ligand that are in the REST region.

        Here the REST region is defined to include combined ligand atoms involved in bond, angle, or improper torsion
        interactions that differ in the end states. Note that proper torsions are omitted from this heuristic as this
        tends to result in larger REST regions than seem desirable.
        """

        aligned_potentials: list[AlignedPotential] = [
            self.aligned_bond,
            self.aligned_angle,
            self.aligned_improper,
        ]

        idxs = {
            int(idx)
            for aligned in aligned_potentials
            for idxs, params_a, params_b in zip(aligned.idxs, aligned.src_params, aligned.dst_params)
            if not np.all(params_a == params_b)
            for idx in idxs  # type: ignore[attr-defined]
        }

        # Ensure all dummy atoms are included in the REST region
        idxs |= self.get_dummy_atoms_a()
        idxs |= self.get_dummy_atoms_b()

        return idxs

    @cached_property
    def rest_region_atom_idxs(self) -> set[int]:
        mol_a_idxs, mol_b_idxs = self.split_combined_idxs(self.base_rest_region_atom_idxs)

        expanded_set_a = self.expand_rest_region_in_mol(mol_a_idxs, self._cycles_a, self.mol_a)
        expanded_set_b = self.expand_rest_region_in_mol(mol_b_idxs, self._cycles_b, self.mol_b)

        final_idxs = set([self.a_to_c[x] for x in expanded_set_a]).union([self.b_to_c[x] for x in expanded_set_b])

        return final_idxs

    @cached_property
    def aliphatic_ring_bonds(self) -> set[CanonicalBond]:
        """Returns the set of aliphatic ring bonds in the combined ligand."""
        ring_bonds_a = {bond.translate(self.a_to_c) for bond in get_aliphatic_ring_bonds(self.mol_a)}
        ring_bonds_b = {bond.translate(self.b_to_c) for bond in get_aliphatic_ring_bonds(self.mol_b)}
        ring_bonds_c = ring_bonds_a | ring_bonds_b
        return ring_bonds_c

    @cached_property
    def rotatable_bonds(self) -> set[CanonicalBond]:
        """Returns the set of rotatable bonds in the combined ligand."""
        rotatable_bonds_a = {bond.translate(self.a_to_c) for bond in get_rotatable_bonds(self.mol_a)}
        rotatable_bonds_b = {bond.translate(self.b_to_c) for bond in get_rotatable_bonds(self.mol_b)}
        rotatable_bonds_c = rotatable_bonds_a | rotatable_bonds_b
        return rotatable_bonds_c

    @cached_property
    def propers(self) -> list[CanonicalProper]:
        """Returns a list of proper torsions in the combined ligand."""
        # TODO: refactor SingleTopology to compute src and dst alignment at initialization
        return [mkproper(*idxs) for idxs in super().setup_intermediate_state(0.0).proper.potential.idxs]

    @cached_property
    def candidate_propers(self) -> dict[int, CanonicalProper]:
        """Returns a dict of propers in the combined ligand, keyed on index, that are candidates for softening."""
        return {
            idx: proper
            for idx, proper in enumerate(self.propers)
            for bond in [mkbond(proper.j, proper.k)]
            if bond in self.rotatable_bonds or bond in self.aliphatic_ring_bonds
        }

    @cached_property
    def target_propers(self) -> dict[int, CanonicalProper]:
        """Returns a dict of propers in the combined ligand, keyed on index, that are candidates for softening and
        involve an atom in the REST region."""
        return {
            idx: proper
            for (idx, proper) in self.candidate_propers.items()
            if any(idx in self.rest_region_atom_idxs for idx in proper.idxs)
        }

    @cached_property
    def target_proper_idxs(self) -> list[int]:
        """Returns a list of indices of propers in the combined ligand that are candidates for softening and involve an
        atom in the REST region."""
        return list(self.target_propers.keys())

    def get_energy_scale_factor(self, lamb: float) -> float:
        temperature_factor = float(self._temperature_scale_interpolation_fxn(lamb))
        return 1.0 / temperature_factor

    def setup_intermediate_state(self, lamb: float) -> GuestSystem:
        ref_state = super().setup_intermediate_state(lamb)
        energy_scale = self.get_energy_scale_factor(lamb)

        assert ref_state.proper
        proper = replace(
            ref_state.proper,
            params=jnp.asarray(ref_state.proper.params).at[self.target_proper_idxs, 0].mul(energy_scale),
        )

        rest_region_pair_idxs = [
            idx
            for idx, (i, j) in enumerate(ref_state.nonbonded_pair_list.potential.idxs)
            if i in self.rest_region_atom_idxs or j in self.rest_region_atom_idxs
        ]

        nonbonded_pair_list = replace(
            ref_state.nonbonded_pair_list,
            params=jnp.asarray(ref_state.nonbonded_pair_list.params)
            .at[rest_region_pair_idxs, NBParamIdx.Q_IDX]
            .mul(energy_scale)  # scale q_ij
            .at[rest_region_pair_idxs, NBParamIdx.LJ_EPS_IDX]
            .mul(energy_scale),  # scale eps_ij
        )

        return replace(ref_state, proper=proper, nonbonded_pair_list=nonbonded_pair_list)

    def combine_with_host(
        self,
        host_system: HostSystem,
        lamb: float,
        num_water_atoms: int,
        ff: Forcefield,
        omm_topology: app.topology.Topology,
    ) -> HostGuestSystem:
        ref_state = super().combine_with_host(host_system, lamb, num_water_atoms, ff, omm_topology)

        # compute indices corresponding to REST-region ligand atoms in the host-guest interaction potential
        num_atoms_host = host_system.nonbonded_all_pairs.potential.num_atoms
        rest_region_atom_idxs = np.array(sorted(self.rest_region_atom_idxs)) + num_atoms_host

        # NOTE: the following methods of scaling the ligand-environment interaction energy are all equivalent:
        #
        # 1. scaling ligand charges and LJ epsilons by energy_scale
        # 2. scaling environment charges and LJ epsilons by energy_scale
        # 3. scaling all charges and LJ epsilons by sqrt(energy_scale)
        #
        # However, (2) and (3) are incompatible with the current water sampling implementation, which assumes that the
        # parameters corresponding to water atoms are identical in the host-host all-pairs potential and the host-guest
        # interaction group potential. Therefore we choose option (1).

        energy_scale = self.get_energy_scale_factor(lamb)

        nonbonded_host_guest_ixn = replace(
            ref_state.nonbonded_ixn_group,
            params=jnp.asarray(ref_state.nonbonded_ixn_group.params)
            .at[rest_region_atom_idxs, NBParamIdx.Q_IDX]
            .mul(energy_scale)  # scale ligand charges
            .at[rest_region_atom_idxs, NBParamIdx.LJ_EPS_IDX]
            .mul(energy_scale),  # scale ligand epsilons
        )

        return replace(ref_state, nonbonded_ixn_group=nonbonded_host_guest_ixn)
