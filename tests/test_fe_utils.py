import numpy as np
import pytest
from rdkit import Chem

from timemachine.fe import utils
from timemachine.fe.model_utils import image_molecule, verify_rabfe_pair


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


def test_image_molecules():
    suppl = Chem.SDMolSupplier("tests/data/benzene_fluorinated.sdf", removeHs=False)
    all_mols = [x for x in suppl]
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


def test_verify_rabfe_pair():
    hydrogen_less_mol = Chem.MolFromSmiles("c1ccccc1")
    blocker_mol = Chem.AddHs(hydrogen_less_mol)

    with pytest.raises(AssertionError) as e:
        verify_rabfe_pair(hydrogen_less_mol, blocker_mol)
    assert "Hydrogens missing for mol" in str(e.value)

    # Verify ordering doesn't matter to pick up missing hydrogens
    with pytest.raises(AssertionError) as e:
        verify_rabfe_pair(blocker_mol, hydrogen_less_mol)
    assert "Hydrogens missing for mol" in str(e.value)

    ligand = Chem.AddHs(hydrogen_less_mol)
    verify_rabfe_pair(ligand, blocker_mol)

    charged_ligand = Chem.AddHs(Chem.MolFromSmiles("C[n+]1cc[nH]c1"))
    with pytest.raises(AssertionError) as e:
        verify_rabfe_pair(charged_ligand, blocker_mol)
    err_msg = str(e.value)
    assert err_msg.startswith("Formal charge disagrees:")
    assert "ligand: 1" in err_msg
