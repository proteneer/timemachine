from dataclasses import replace
from functools import cached_property
from typing import Callable, Literal

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from openmm import app

from timemachine.constants import NBParamIdx
from timemachine.fe.single_topology import SingleTopology
from timemachine.fe.system import HostGuestSystem, VacuumSystem
from timemachine.ff import Forcefield
from timemachine.potentials import HarmonicAngleStable, NonbondedPairListPrecomputed

from .bond import CanonicalBond, mkbond
from .interpolation import Exponential, InterpolationFxn, Linear, Quadratic, Symmetric
from .queries import get_rotatable_bonds

InterpolationFxnName = Literal["linear", "quadratic", "exponential"]


def get_temperature_scale_interpolation_fxn(
    max_temperature_scale: float, interpolation_fxn: InterpolationFxnName
) -> InterpolationFxn:
    make_interp_fxn: Callable[[float, float], InterpolationFxn]
    match interpolation_fxn:
        case "linear":
            make_interp_fxn = Linear
        case "quadratic":
            make_interp_fxn = Quadratic
        case "exponential":
            make_interp_fxn = Exponential

    return Symmetric(make_interp_fxn(1.0, max_temperature_scale))


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

    Parameters
    ----------
    mol_a: ROMol
        First guest

    mol_b: ROMol
        Second guest

    core: np.array (C, 2)
        Atom mapping from mol_a to mol_b

    forcefield: ff.Forcefield
        Forcefield to use for parameterization

    max_temperature_scale: float
        Maximum temperature scale factor

    temperature_scale_interpolation_fxn: str
        Interpolation function to use for temperature scaling. One of "linear", "quadratic", or "exponential"
    """

    def __init__(
        self,
        mol_a,
        mol_b,
        core,
        forcefield,
        max_temperature_scale: float,
        temperature_scale_interpolation_fxn: InterpolationFxnName = "exponential",
    ):
        super().__init__(mol_a, mol_b, core, forcefield)

        self._temperature_scale_interpolation_fxn = get_temperature_scale_interpolation_fxn(
            max_temperature_scale, temperature_scale_interpolation_fxn
        )

    @cached_property
    def aliphatic_ring_bonds(self) -> set[CanonicalBond]:
        def get_aliphatic_ring_bonds(mol):
            return [
                mkbond(
                    mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx(),
                    mol.GetBondWithIdx(bond_idx).GetEndAtomIdx(),
                )
                for ring_bond_idxs in mol.GetRingInfo().BondRings()
                for is_aromatic in [all(mol.GetBondWithIdx(bond_idx).GetIsAromatic() for bond_idx in ring_bond_idxs)]
                if not is_aromatic
                for bond_idx in ring_bond_idxs
            ]

        ring_bonds_a = {bond.translate(self.a_to_c) for bond in get_aliphatic_ring_bonds(self.mol_a)}
        ring_bonds_b = {bond.translate(self.b_to_c) for bond in get_aliphatic_ring_bonds(self.mol_b)}
        ring_bonds_c = ring_bonds_a | ring_bonds_b
        return ring_bonds_c

    @cached_property
    def rotatable_bonds(self) -> set[CanonicalBond]:
        rotatable_bonds_a = {bond.translate(self.a_to_c) for bond in get_rotatable_bonds(self.mol_a)}
        rotatable_bonds_b = {bond.translate(self.b_to_c) for bond in get_rotatable_bonds(self.mol_b)}
        rotatable_bonds_c = rotatable_bonds_a | rotatable_bonds_b
        return rotatable_bonds_c

    @cached_property
    def propers(self) -> NDArray[np.int32]:
        # TODO: refactor SingleTopology to compute src and dst alignment at initialization
        return super().setup_intermediate_state(0.0).proper.potential.idxs

    @cached_property
    def target_proper_idxs(self) -> list[int]:
        return [
            idx
            for idx, (_, j, k, _) in enumerate(self.propers)
            for bond in [mkbond(j, k)]
            if bond in self.rotatable_bonds or bond in self.aliphatic_ring_bonds
        ]

    @cached_property
    def target_propers(self) -> NDArray[np.int32]:
        return self.propers[self.target_proper_idxs, :]

    def get_energy_scale_factor(self, lamb: float) -> float:
        temperature_factor = self._temperature_scale_interpolation_fxn(lamb).item()
        return 1.0 / temperature_factor

    def setup_intermediate_state(self, lamb: float) -> VacuumSystem[NonbondedPairListPrecomputed, HarmonicAngleStable]:
        ref_state = super().setup_intermediate_state(lamb)
        energy_scale = self.get_energy_scale_factor(lamb)

        assert ref_state.proper
        proper = replace(
            ref_state.proper,
            params=jnp.asarray(ref_state.proper.params).at[self.target_proper_idxs, 0].mul(energy_scale),
        )

        nonbonded = replace(
            ref_state.nonbonded,
            params=jnp.asarray(ref_state.nonbonded.params)
            .at[:, NBParamIdx.Q_IDX]
            .mul(energy_scale)  # scale q_ij
            .at[:, NBParamIdx.LJ_EPS_IDX]
            .mul(energy_scale),  # scale eps_ij
        )

        return replace(ref_state, proper=proper, nonbonded=nonbonded)

    def combine_with_host(
        self,
        host_system: VacuumSystem,
        lamb: float,
        num_water_atoms: int,
        ff: Forcefield,
        omm_topology: app.topology.Topology,
    ) -> HostGuestSystem:
        ref_state = super().combine_with_host(host_system, lamb, num_water_atoms, ff, omm_topology)
        num_atoms_host = host_system.nonbonded.potential.num_atoms
        energy_scale = self.get_energy_scale_factor(lamb)

        nonbonded_host_guest_ixn = replace(
            ref_state.nonbonded_host_guest_ixn,
            params=jnp.asarray(ref_state.nonbonded_host_guest_ixn.params)
            .at[num_atoms_host:, NBParamIdx.Q_IDX]
            .mul(energy_scale)  # scale ligand charges
            .at[num_atoms_host:, NBParamIdx.LJ_EPS_IDX]
            .mul(energy_scale),  # scale ligand epsilons
        )

        return replace(ref_state, nonbonded_host_guest_ixn=nonbonded_host_guest_ixn)
