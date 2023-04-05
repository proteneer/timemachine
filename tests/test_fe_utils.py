import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine.fe import model_utils, utils
from timemachine.fe.model_utils import image_frame, image_molecule
from timemachine.ff import Forcefield

pytestmark = [pytest.mark.nogpu]


def test_sanitize_energies():

    full_us = np.array([[15000.0, -5081923.0, 1598, 1.5, -23.0], [-423581.0, np.nan, -238, 13.5, 23.0]])

    test_us = utils.sanitize_energies(full_us, lamb_idx=3, cutoff=10000)

    expected_us = np.array([[np.inf, np.inf, 1598, 1.5, -23.0], [np.inf, np.inf, -238, 13.5, 23.0]])

    np.testing.assert_array_equal(test_us, expected_us)


def test_extract_delta_Us_from_U_knk():

    K = 4
    N = 8

    U_knk = np.random.rand(K, N, K)

    test_delta_Us = utils.extract_delta_Us_from_U_knk(U_knk)

    expected_delta_Us = np.array(
        [
            # fwd                          # rev
            [U_knk[0, :, 1] - U_knk[0, :, 0], U_knk[1, :, 0] - U_knk[1, :, 1]],
            [U_knk[1, :, 2] - U_knk[1, :, 1], U_knk[2, :, 1] - U_knk[2, :, 2]],
            [U_knk[2, :, 3] - U_knk[2, :, 2], U_knk[3, :, 2] - U_knk[3, :, 3]],
        ]
    )

    np.testing.assert_almost_equal(expected_delta_Us, test_delta_Us)


def test_image_frame():
    rng = np.random.default_rng(2022)

    coords = rng.random((90, 3))

    max_dimensions = np.max(coords, axis=0)

    # Add a random buffer to the dimensions of the box
    box = np.eye(3) * (max_dimensions + rng.random(max_dimensions.shape))
    idxs = np.arange(len(coords))
    group_indices = []
    group_indices.extend(list(idxs[:30].reshape(-1, 3)))
    group_indices.extend(list(idxs[30:].reshape(-1, 5)))
    group_indices.append(np.array([], dtype=idxs.dtype))

    box_diag = np.diagonal(box)

    imaged_coords = image_frame(group_indices, coords, box)
    np.testing.assert_allclose(coords, imaged_coords)

    for _ in range(100):
        new_coords = coords.copy()

        for group in group_indices:
            # shift each direction randomly, need to go beyond one image since waters can float very far away
            x_shift, y_shift, z_shift = rng.integers(-5, 5, size=3)

            offset = np.array([x_shift * box_diag[0], y_shift * box_diag[1], z_shift * box_diag[2]])
            offset = np.expand_dims(offset, axis=0)  # make this broad castable to [N,3]
            new_coords[group] += offset
        imaged_coords = image_frame(group_indices, new_coords, box)
        np.testing.assert_allclose(coords, imaged_coords)


def test_image_molecules():
    all_mols = utils.read_sdf("tests/data/benzene_fluorinated.sdf")
    mol = all_mols[0]
    mol_coords = utils.get_romol_conf(mol)

    np.random.seed(2022)
    # Shift the mol so that all coords are 0 or positive
    mol_coords += np.abs(np.min(mol_coords, axis=0))
    # Get the max dimensions of the mol to construct the box
    max_dimensions = np.max(mol_coords, axis=0)

    # Add a random buffer to the dimensions of the box
    box = np.eye(3) * (max_dimensions + np.random.random(max_dimensions.shape))

    # If the mol is already in the box, the coordinates will be the same
    np.testing.assert_array_equal(mol_coords, image_molecule(mol_coords, box))
    box_diag = np.diagonal(box)

    for _ in range(1000):
        # shift each direction randomly, need to go beyond one image since waters can float very far away
        x_shift = np.random.randint(-5, 5)
        y_shift = np.random.randint(-5, 5)
        z_shift = np.random.randint(-5, 5)

        offset = np.array([x_shift * box_diag[0], y_shift * box_diag[1], z_shift * box_diag[2]])
        offset = np.expand_dims(offset, axis=0)  # make this broad castable to [N,3]
        new_mol_conf = mol_coords + offset

        imaged_mol = image_molecule(new_mol_conf, box)

        np.testing.assert_array_almost_equal(imaged_mol, mol_coords)


def test_get_mol_name():
    mol = Chem.MolFromSmiles("c1ccccc1")
    with pytest.raises(KeyError):
        utils.get_mol_name(mol)

    mol.SetProp("_Name", "test_name")
    assert utils.get_mol_name(mol) == "test_name"


def test_set_mol_coords():
    np.random.seed(2022)
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(mol)

    x0 = utils.get_romol_conf(mol)

    # Make some random move
    x1 = x0 + np.random.randn(*x0.shape)

    # This is lossy
    for precision in [np.float32, np.float64]:
        utils.set_romol_conf(mol, x1.astype(precision))

        x1_copy = utils.get_romol_conf(mol)

        # Won't be exact, but should be close
        assert not np.all(x1 == x1_copy)
        np.testing.assert_allclose(x1, x1_copy)


def test_experimental_conversions_to_kj():
    rng = np.random.RandomState(2022)

    experimental_values = rng.random(10)
    # Verify that uM to kJ and uIC50 to Kj is identical
    np.testing.assert_array_equal(
        utils.convert_uM_to_kJ_per_mole(experimental_values), utils.convert_uIC50_to_kJ_per_mole(experimental_values)
    )

    np.testing.assert_allclose(utils.convert_uM_to_kJ_per_mole(0.15), -38.951164)


def test_get_strained_atoms():
    ff = Forcefield.load_default()
    np.random.seed(2022)
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(mol)
    assert model_utils.get_strained_atoms(mol, ff) == []

    # force a clash
    x0 = utils.get_romol_conf(mol)
    x0[-2, :] = x0[-1, :] + 0.01
    utils.set_romol_conf(mol, x0)
    assert model_utils.get_strained_atoms(mol, ff) == [10, 11]
