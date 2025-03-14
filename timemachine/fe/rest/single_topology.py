from dataclasses import replace
from functools import cached_property

import jax.numpy as jnp
from numpy.typing import NDArray
from openmm import app
from rdkit import Chem

from timemachine.constants import NBParamIdx
from timemachine.fe.single_topology import SingleTopology
from timemachine.fe.system import GuestSystem, HostGuestSystem, HostSystem
from timemachine.ff import Forcefield

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

        self._temperature_scale_interpolation_fxn: InterpolationFxn = get_temperature_scale_interpolation_fxn(
            max_temperature_scale, temperature_scale_interpolation
        )

    @cached_property
    def rest_region_atom_idxs(self) -> set[int]:
        """Returns the set of indices of atoms in the combined ligand that are in the REST region."""
        return {
            idx
            for diffs in [
                self.aligned_bond_tuples,
                self.aligned_angle_tuples,
                self.aligned_proper_tuples,
                self.aligned_improper_tuples,
                self.aligned_chiral_atom_tuples,
                # self.aligned_chiral_bond_tuples, # NOTE: chiral bonds not implemented
                self.aligned_nonbonded_pairlist_tuples,
            ]
            for diff_idxs, _, _ in diffs
            for idx in diff_idxs
        }

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
    def target_proper_idxs(self) -> list[int]:
        """Returns a list of indices of the proper torsion interactions in the combined ligand that are targeted for
        softening."""
        return [
            idx
            for idx, proper in enumerate(self.propers)
            for bond in [mkbond(proper.j, proper.k)]
            if bond in self.rotatable_bonds or bond in self.aliphatic_ring_bonds
        ]

    @cached_property
    def target_propers(self) -> set[CanonicalProper]:
        """Returns the set of proper torsions in the combined ligand that are targeted for softening."""
        return {self.propers[i] for i in self.target_proper_idxs}

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

        nonbonded_pair_list = replace(
            ref_state.nonbonded_pair_list,
            params=jnp.asarray(ref_state.nonbonded_pair_list.params)
            .at[:, NBParamIdx.Q_IDX]
            .mul(energy_scale)  # scale q_ij
            .at[:, NBParamIdx.LJ_EPS_IDX]
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

        num_atoms_host = host_system.nonbonded_all_pairs.potential.num_atoms
        ligand_idxs = slice(num_atoms_host, None, None)

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
            .at[ligand_idxs, NBParamIdx.Q_IDX]
            .mul(energy_scale)  # scale ligand charges
            .at[ligand_idxs, NBParamIdx.LJ_EPS_IDX]
            .mul(energy_scale),  # scale ligand epsilons
        )

        return replace(ref_state, nonbonded_ixn_group=nonbonded_host_guest_ixn)
