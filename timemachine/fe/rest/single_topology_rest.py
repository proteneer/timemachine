from dataclasses import replace
from functools import cached_property

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from openmm import app

from timemachine.fe.single_topology import SingleTopology
from timemachine.fe.system import HostGuestSystem, VacuumSystem
from timemachine.ff import Forcefield
from timemachine.md.enhanced import identify_rotatable_bonds
from timemachine.potentials import HarmonicAngleStable, NonbondedPairListPrecomputed

from .bond import CanonicalBond, mkbond
from .interpolation import InterpolationFxn, Symmetric


class SingleTopologyREST(SingleTopology):
    def __init__(
        self,
        mol_a,
        mol_b,
        core,
        forcefield,
        temperature_scale_interpolation_fxn: Symmetric[InterpolationFxn],
    ):
        assert isinstance(temperature_scale_interpolation_fxn, Symmetric)
        assert temperature_scale_interpolation_fxn.src == 1.0

        super().__init__(mol_a, mol_b, core, forcefield)

        self._temperature_scale_interpolation_fxn = temperature_scale_interpolation_fxn

    @cached_property
    def aliphatic_ring_bonds(self):
        def get_aliphatic_ring_bonds(mol):
            return [
                (
                    mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx(),
                    mol.GetBondWithIdx(bond_idx).GetEndAtomIdx(),
                )
                for ring_bond_idxs in mol.GetRingInfo().BondRings()
                for is_aromatic in [all(mol.GetBondWithIdx(bond_idx).GetIsAromatic() for bond_idx in ring_bond_idxs)]
                if not is_aromatic
                for bond_idx in ring_bond_idxs
            ]

        ring_bonds_a = {mkbond(i, j).translate(self.a_to_c) for i, j in get_aliphatic_ring_bonds(self.mol_a)}
        ring_bonds_b = {mkbond(i, j).translate(self.b_to_c) for i, j in get_aliphatic_ring_bonds(self.mol_b)}
        ring_bonds_c = ring_bonds_a | ring_bonds_b
        return ring_bonds_c

    @cached_property
    def rotatable_bonds(self) -> set[CanonicalBond]:
        rotatable_bonds_a = {mkbond(i, j).translate(self.a_to_c) for i, j in identify_rotatable_bonds(self.mol_a)}
        rotatable_bonds_b = {mkbond(i, j).translate(self.b_to_c) for i, j in identify_rotatable_bonds(self.mol_b)}
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
            .at[:, 0]
            .mul(energy_scale)  # scale q_ij
            .at[:, 2]
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
            .at[num_atoms_host:, 0]
            .mul(energy_scale)  # scale ligand charges
            .at[num_atoms_host:, 2]
            .mul(energy_scale),  # scale ligand epsilons
        )

        return replace(ref_state, nonbonded_host_guest_ixn=nonbonded_host_guest_ixn)
