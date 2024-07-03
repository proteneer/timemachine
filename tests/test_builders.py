from importlib import resources

import numpy as np

from timemachine.constants import DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF
from timemachine.ff.handlers import openmm_deserializer
from timemachine.md.builders import build_protein_system, build_water_system
from timemachine.md.minimizer import check_force_norm
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


def test_build_water_system():
    mol_a, mol_b, _ = get_hif2a_ligand_pair_single_topology()

    water_system, water_coords, box, _ = build_water_system(4.0, DEFAULT_WATER_FF)

    water_with_mols, mol_water_coords, box_with_mols, _ = build_water_system(4.0, DEFAULT_WATER_FF, mols=[mol_a, mol_b])

    # A few waters should have been deleted when adding in the molecules
    assert len(water_coords) > len(mol_water_coords)
    np.testing.assert_equal(box, box_with_mols)

    box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes

    water_system_bps, _ = openmm_deserializer.deserialize_system(water_system, cutoff=1.2)
    for bp in water_system_bps:
        (
            du_dx,
            _,
        ) = bp.to_gpu(
            np.float32
        ).bound_impl.execute(water_coords, box, compute_u=False)
        check_force_norm(-du_dx)

    water_system_bps, _ = openmm_deserializer.deserialize_system(water_with_mols, cutoff=1.2)
    for bp in water_system_bps:
        (
            du_dx,
            _,
        ) = bp.to_gpu(
            np.float32
        ).bound_impl.execute(mol_water_coords, box, compute_u=False)
        check_force_norm(-du_dx)


def test_build_protein_system():
    mol_a, mol_b, _ = get_hif2a_ligand_pair_single_topology()

    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as pdb_path:
        host_pdbfile = str(pdb_path)
    protein_system, protein_coords, box, _, num_water_atoms = build_protein_system(
        host_pdbfile, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF
    )
    num_host_atoms = protein_coords.shape[0] - num_water_atoms

    protein_with_mols, mol_protein_coords, box_with_mols, _, num_water_atoms_with_mols = build_protein_system(
        host_pdbfile, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, mols=[mol_a, mol_b]
    )
    num_host_atoms_with_mol = mol_protein_coords.shape[0] - num_water_atoms_with_mols

    assert num_host_atoms == num_host_atoms_with_mol
    # Waters won't be deleted since the pocket has no waters
    assert num_water_atoms == num_water_atoms_with_mols
    np.testing.assert_equal(box, box_with_mols)

    box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes

    protein_system_bps, _ = openmm_deserializer.deserialize_system(protein_system, cutoff=1.2)
    for bp in protein_system_bps:
        (
            du_dx,
            _,
        ) = bp.to_gpu(
            np.float32
        ).bound_impl.execute(protein_coords, box, compute_u=False)
        check_force_norm(-du_dx)

    protein_system_bps, _ = openmm_deserializer.deserialize_system(protein_with_mols, cutoff=1.2)
    for bp in protein_system_bps:
        (
            du_dx,
            _,
        ) = bp.to_gpu(
            np.float32
        ).bound_impl.execute(mol_protein_coords, box, compute_u=False)
        check_force_norm(-du_dx)
