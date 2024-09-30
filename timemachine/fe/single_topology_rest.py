from dataclasses import replace

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray
from openmm import app

from timemachine.fe.single_topology import AtomMapMixin, SingleTopology
from timemachine.fe.system import HostGuestSystem, VacuumSystem
from timemachine.ff import Forcefield
from timemachine.md.enhanced import identify_rotatable_bonds


def get_rest_temperature_factor(lamb: ArrayLike, max_value: float = 2.0) -> NDArray:
    def f(x):
        return (1.0 - x) + x * max_value

    lamb = np.asarray(lamb)
    return np.where(
        lamb < 0.5,
        f(2.0 * lamb),
        f(2.0 * (1.0 - lamb)),
    )


def translate_bond(bond, a_to_b):
    return tuple(a_to_b[idx] for idx in bond)


def get_rotatable_bonds(mol_a, mol_b, core):
    atom_map = AtomMapMixin(mol_a, mol_b, core)
    rotatable_bonds_a = {translate_bond(bond, atom_map.a_to_c) for bond in identify_rotatable_bonds(mol_a)}
    rotatable_bonds_b = {translate_bond(bond, atom_map.b_to_c) for bond in identify_rotatable_bonds(mol_b)}
    rotatable_bonds_c = rotatable_bonds_a | rotatable_bonds_b
    return rotatable_bonds_c


def get_non_aromatic_ring_bonds(mol_a, mol_b, core):
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


def scale_torsion(torsion, target_torsions, scale):
    return replace(torsion, params=jnp.asarray(torsion.params).at[target_torsions, 0].mul(scale))


def scale_nonbonded_pairs(nonbonded_pairs, scale):
    return replace(
        nonbonded_pairs,
        params=jnp.asarray(nonbonded_pairs.params)
        .at[:, 0]
        .mul(scale)  # scale q_ij
        .at[:, 2]
        .mul(scale),  # scale eps_ij
    )


def scale_nonbonded_host_guest_ixn(nonbonded_host_guest_ixn, num_atoms_host, scale):
    return replace(
        nonbonded_host_guest_ixn,
        params=jnp.asarray(nonbonded_host_guest_ixn.params)
        .at[num_atoms_host:, 0]
        .mul(scale)  # scale ligand charges
        .at[num_atoms_host:, 2]
        .mul(scale),  # scale ligand epsilons
    )


class SingleTopologyREST(SingleTopology):
    def __init__(self, *args, max_temperature_factor: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_temperature_factor = max_temperature_factor
        self._rest_torsions = self._get_rest_torsions()

    def _get_rest_torsions(self) -> NDArray:
        rotatable_bonds = get_rotatable_bonds(self.mol_a, self.mol_b, self.core)
        ring_bonds = get_non_aromatic_ring_bonds(self.mol_a, self.mol_b, self.core)
        target_bonds = rotatable_bonds | ring_bonds

        assert self.src_system.torsion
        torsions = self.src_system.torsion.potential.idxs
        target_torsions = [(i, j, k, l) for (i, j, k, l) in torsions if (j, k) in target_bonds]
        return np.array(target_torsions)

    def get_rest_energy_scale_factor(self, lamb: float):
        temperature_factor = get_rest_temperature_factor(lamb, self._max_temperature_factor)
        return 1.0 / temperature_factor

    def setup_intermediate_state(self, lamb: float) -> VacuumSystem:
        ref_state = super().setup_intermediate_state(lamb)
        scale = self.get_rest_energy_scale_factor(lamb)
        assert ref_state.torsion
        rest_torsion_mask = (ref_state.torsion.potential.idxs[:, None] == self._rest_torsions[None, :]).all(-1).any(-1)
        return replace(
            ref_state,
            torsion=scale_torsion(ref_state.torsion, rest_torsion_mask, scale),
            nonbonded=scale_nonbonded_pairs(ref_state.nonbonded, scale),
        )

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
        return replace(
            ref_state,
            nonbonded_host_guest_ixn=scale_nonbonded_host_guest_ixn(
                ref_state.nonbonded_host_guest_ixn, num_atoms_host, scale
            ),
        )
