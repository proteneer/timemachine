from dataclasses import replace
from typing import Literal, Set, Tuple

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray
from openmm import app

from timemachine.fe.bond import CanonicalBond, CanonicalBonds
from timemachine.fe.single_topology import AtomMapMixin, SingleTopology
from timemachine.fe.system import HostGuestSystem, VacuumSystem
from timemachine.ff import Forcefield
from timemachine.md.enhanced import identify_rotatable_bonds
from timemachine.potentials.potentials import (
    BoundPotential,
    HarmonicAngle,
    HarmonicAngleStable,
    NonbondedInteractionGroup,
    NonbondedPairListPrecomputed,
    PeriodicTorsion,
)


def get_rest_temperature_factor(
    lamb: ArrayLike, max_value: float, schedule: Literal["linear", "quadratic", "exponential"]
) -> NDArray[np.float64]:
    def f(x):
        match schedule:
            case "linear":
                return (1.0 - x) + x * max_value
            case "quadratic":
                return 1.0 + (max_value - 1.0) * x**2
            case "exponential":
                return np.exp(np.log(max_value) * x)

    lamb = np.asarray(lamb)
    return np.where(
        lamb < 0.5,
        f(2.0 * lamb),
        f(2.0 * (1.0 - lamb)),
    )


def translate_bond(bond: Tuple[int, int], a_to_b: NDArray[np.int32]) -> CanonicalBond:
    i, j = bond
    return CanonicalBond.from_tuple((int(a_to_b[i]), int(a_to_b[j])))


def get_rotatable_bonds(mol_a, mol_b, core) -> Set[CanonicalBond]:
    atom_map = AtomMapMixin(mol_a, mol_b, core)
    rotatable_bonds_a = {translate_bond(bond, atom_map.a_to_c) for bond in identify_rotatable_bonds(mol_a)}
    rotatable_bonds_b = {translate_bond(bond, atom_map.b_to_c) for bond in identify_rotatable_bonds(mol_b)}
    rotatable_bonds_c = rotatable_bonds_a | rotatable_bonds_b
    return rotatable_bonds_c


def get_non_aromatic_ring_bonds(mol_a, mol_b, core) -> Set[CanonicalBond]:
    def get_ring_bonds_mol(mol):
        return [
            (
                mol.GetBondWithIdx(idx).GetBeginAtomIdx(),
                mol.GetBondWithIdx(idx).GetEndAtomIdx(),
            )
            for ring in mol.GetRingInfo().BondRings()
            if not any(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
            for idx in ring
        ]

    atom_map = AtomMapMixin(mol_a, mol_b, core)
    ring_bonds_a = {translate_bond(bond, atom_map.a_to_c) for bond in get_ring_bonds_mol(mol_a)}
    ring_bonds_b = {translate_bond(bond, atom_map.b_to_c) for bond in get_ring_bonds_mol(mol_b)}
    ring_bonds_c = ring_bonds_a | ring_bonds_b
    return ring_bonds_c


def ixn_isin(test_ixns: CanonicalBonds, ixns: CanonicalBonds):
    eq = test_ixns.idxs[:, None] == ixns.idxs[None, :]
    return eq.all(-1).any(-1)


def scale_angle(angle: BoundPotential[HarmonicAngle], target_angles: CanonicalBonds, scale: ArrayLike):
    assert target_angles.idxs.shape[1] == 3
    target_angles = ixn_isin(CanonicalBonds.from_idxs(angle.potential.idxs, 3), target_angles)
    return replace(angle, params=jnp.asarray(angle.params).at[target_angles, 0].mul(scale))


def scale_torsion(torsion: BoundPotential[PeriodicTorsion], target_torsions: CanonicalBonds, scale: ArrayLike):
    assert target_torsions.idxs.shape[1] == 4
    target_torsions = ixn_isin(CanonicalBonds.from_idxs(torsion.potential.idxs, 4), target_torsions)
    return replace(torsion, params=jnp.asarray(torsion.params).at[target_torsions, 0].mul(scale))


def scale_nonbonded_pairs(nonbonded_pairs: BoundPotential[HarmonicAngle], scale: ArrayLike):
    return replace(
        nonbonded_pairs,
        params=jnp.asarray(nonbonded_pairs.params)
        .at[:, 0]
        .mul(scale)  # scale q_ij
        .at[:, 2]
        .mul(scale),  # scale eps_ij
    )


def scale_nonbonded_host_guest_ixn(
    nonbonded_host_guest_ixn: BoundPotential[NonbondedInteractionGroup], num_atoms_host: int, scale: ArrayLike
):
    return replace(
        nonbonded_host_guest_ixn,
        params=jnp.asarray(nonbonded_host_guest_ixn.params)
        .at[num_atoms_host:, 0]
        .mul(scale)  # scale ligand charges
        .at[num_atoms_host:, 2]
        .mul(scale),  # scale ligand epsilons
    )


TemperatureSchedule = Literal["linear", "quadratic", "exponential"]


class SingleTopologyREST(SingleTopology):
    def __init__(
        self,
        mol_a,
        mol_b,
        core,
        forcefield,
        max_temperature_factor: float,
        temperature_schedule: TemperatureSchedule,
        scale_angles: bool,
    ):
        super().__init__(mol_a, mol_b, core, forcefield)

        self._max_temperature_factor = max_temperature_factor
        self._temperature_schedule: TemperatureSchedule = temperature_schedule
        self._scale_angles = scale_angles

        self._rest_angles = self._get_rest_angles()
        self._rest_torsions = self._get_rest_torsions()

    def _get_rest_angles(self) -> CanonicalBonds:
        ring_bonds = get_non_aromatic_ring_bonds(self.mol_a, self.mol_b, self.core)
        angles = self.src_system.angle.potential.idxs
        target_angles = [
            (i, j, k)
            for (i, j, k) in angles
            if CanonicalBond.from_tuple((i, j)) in ring_bonds and CanonicalBond.from_tuple((j, k)) in ring_bonds
        ]
        return CanonicalBonds.from_idxs(target_angles, 3)

    def _get_rest_torsions(self) -> CanonicalBonds:
        rotatable_bonds = get_rotatable_bonds(self.mol_a, self.mol_b, self.core)
        non_aromatic_ring_bonds = get_non_aromatic_ring_bonds(self.mol_a, self.mol_b, self.core)

        def is_rotatable_bond(j, k):
            return CanonicalBond.from_tuple((j, k)) in rotatable_bonds

        def is_non_aromatic_ring_torsion(i, j, k, l):
            return {
                CanonicalBond.from_tuple((i, j)),
                CanonicalBond.from_tuple((j, k)),
                CanonicalBond.from_tuple((k, l)),
            }.issubset(non_aromatic_ring_bonds)

        assert self.src_system.torsion
        torsions = self.src_system.torsion.potential.idxs
        target_torsions = [
            (i, j, k, l)
            for (i, j, k, l) in torsions
            if is_rotatable_bond(j, k) or is_non_aromatic_ring_torsion(i, j, k, l)
        ]
        return CanonicalBonds.from_idxs(target_torsions, 4)

    def get_rest_energy_scale_factor(self, lamb: float) -> NDArray[np.float64]:
        temperature_factor = get_rest_temperature_factor(lamb, self._max_temperature_factor, self._temperature_schedule)
        return 1.0 / temperature_factor

    def setup_intermediate_state(self, lamb: float) -> VacuumSystem[NonbondedPairListPrecomputed, HarmonicAngleStable]:
        ref_state = super().setup_intermediate_state(lamb)
        scale = self.get_rest_energy_scale_factor(lamb)

        angle = scale_angle(ref_state.angle, self._rest_angles, scale) if self._scale_angles else ref_state.angle

        assert ref_state.torsion
        torsion = scale_torsion(ref_state.torsion, self._rest_torsions, scale)

        nonbonded = scale_nonbonded_pairs(ref_state.nonbonded, scale)

        return replace(ref_state, angle=angle, torsion=torsion, nonbonded=nonbonded)

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
        scale = self.get_rest_energy_scale_factor(lamb)

        nonbonded_host_guest_ixn = scale_nonbonded_host_guest_ixn(
            ref_state.nonbonded_host_guest_ixn, num_atoms_host, scale
        )

        return replace(ref_state, nonbonded_host_guest_ixn=nonbonded_host_guest_ixn)
