from importlib import resources

import numpy as np

from timemachine.constants import DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF
from timemachine.fe.utils import get_romol_conf, set_romol_conf
from timemachine.ff.handlers import openmm_deserializer
from timemachine.md.barostat.utils import compute_box_volume
from timemachine.md.builders import build_protein_system, build_water_system
from timemachine.md.minimizer import check_force_norm
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


def test_build_water_system():
    mol_a, mol_b, _ = get_hif2a_ligand_pair_single_topology()

    water_system, water_coords, box, _ = build_water_system(4.0, DEFAULT_WATER_FF)

    water_with_mols, mol_water_coords, box_with_mols, _ = build_water_system(4.0, DEFAULT_WATER_FF, mols=[mol_a, mol_b])

    # No waters should be deleted, but the box will be slightly larger
    assert len(water_coords) == len(mol_water_coords)
    assert compute_box_volume(box) < compute_box_volume(box_with_mols)

    box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    box_with_mols += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes

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
        ).bound_impl.execute(mol_water_coords, box_with_mols, compute_u=False)
        check_force_norm(-du_dx)


def test_build_protein_system():
    rng = np.random.default_rng(2024)
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
    np.testing.assert_equal(compute_box_volume(box), compute_box_volume(box_with_mols))

    box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    box_with_mols += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes

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

    # Pick a random water atom, will center the ligands on the atom and verify that the box is slightly
    # larger
    water_atom_idx = rng.choice(num_water_atoms_with_mols)
    new_ligand_center = mol_protein_coords[num_host_atoms_with_mol + water_atom_idx]
    for mol in [mol_a, mol_b]:
        conf = get_romol_conf(mol)
        centroid = np.mean(conf, axis=0)
        conf = conf - centroid + new_ligand_center
        set_romol_conf(mol, conf)
    _, moved_conf, moved_box, _, num_water_after_moved = build_protein_system(
        host_pdbfile, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, mols=[mol_a, mol_b]
    )
    assert num_water_after_moved == num_water_atoms_with_mols
    host_atoms_with_moved_ligands = moved_conf.shape[0] - num_water_after_moved
    assert num_host_atoms == host_atoms_with_moved_ligands
    assert compute_box_volume(box) < compute_box_volume(moved_box)
