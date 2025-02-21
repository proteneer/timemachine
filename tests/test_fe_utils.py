import numpy as np
import py3Dmol
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine.constants import DEFAULT_TEMP
from timemachine.fe import model_utils, utils
from timemachine.fe.model_utils import image_frame, image_molecule
from timemachine.ff import Forcefield
from timemachine.md.barostat.utils import get_group_indices
from timemachine.potentials import bonded, nonbonded
from timemachine.utils import path_to_internal_file

pytestmark = [pytest.mark.nocuda]


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


def test_image_frame_energy_invariance():
    # test that calling image frames does not affect the energies of bonded and nonbonded terms.
    np.random.seed(2023)

    # box of size [5,4,3]
    box = np.eye(3)
    box[0][0] = 5
    box[1][1] = 4
    box[2][2] = 3
    n_atoms = 500  # num atoms
    n_bonds = 100  # num bonds

    # generate coordinates between [-10, +10]
    old_coords = (np.random.rand(n_atoms, 3) - 0.5) * 20
    nb_params = np.random.rand(n_atoms, 4)
    nb_params[:, 0] -= 0.5  # make charges pos and neg
    nb_params[:, 1] *= 0.1  # decrease sigma to avoid singularities
    nb_params[:, -1] *= 0.1  # vary 4D decoupling

    exc_idxs = np.array([], dtype=np.int32).reshape(0, 2)
    exc_sfs = np.array([], dtype=np.float64).reshape(0, 2)

    bond_idxs = []
    atom_idxs = np.arange(n_atoms)
    # Get a list of atoms to make up the first element of the bonds, then increment by one to ensure consecutive
    bond_atoms = np.random.choice(atom_idxs[:-1], size=n_bonds, replace=False)
    for first_bond in bond_atoms:
        bond_idxs.append([first_bond, first_bond + 1])

    group_idxs = get_group_indices(bond_idxs, n_atoms)
    new_coords = image_frame(group_idxs, old_coords, box)

    # assert centroids are in box

    for atom_idxs in group_idxs:
        x, y, z = np.mean(new_coords[atom_idxs], axis=0)
        assert x > 0 and x < box[0][0]
        assert y > 0 and y < box[1][1]
        assert z > 0 and z < box[2][2]

    assert not np.array_equal(old_coords, new_coords)

    old_nb_U = nonbonded.nonbonded(old_coords, nb_params, box, exc_idxs, exc_sfs, 1.2, 1.3)
    new_nb_U = nonbonded.nonbonded(new_coords, nb_params, box, exc_idxs, exc_sfs, 1.2, 1.3)

    np.testing.assert_allclose(old_nb_U, new_nb_U)

    hb_params = np.random.rand(n_bonds, 2)
    bond_idxs = np.array(bond_idxs)
    old_hb_U = bonded.harmonic_bond(old_coords, hb_params, box, bond_idxs)
    new_hb_U = bonded.harmonic_bond(new_coords, hb_params, box, bond_idxs)

    np.testing.assert_allclose(old_hb_U, new_hb_U)


def test_image_molecules():
    with path_to_internal_file("timemachine.testsystems.data", "benzene_fluorinated.sdf") as path_to_sdf:
        all_mols = utils.read_sdf(path_to_sdf)
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


def test_get_and_set_mol_name():
    mol = Chem.MolFromSmiles("c1ccccc1")
    with pytest.raises(KeyError):
        utils.get_mol_name(mol)

    utils.set_mol_name(mol, "test_name")
    assert utils.get_mol_name(mol) == "test_name"


def test_get_and_set_mol_coords():
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


def test_get_and_set_mol_coords_conf_id():
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMultipleConfs(mol, numConfs=2, randomSeed=2024)

    x0 = utils.get_romol_conf(mol, 0)
    x1 = utils.get_romol_conf(mol, 1)
    assert np.any(x0 != x1)  # expect different confs

    utils.set_romol_conf(mol, x0, 1)
    x1_new = utils.get_romol_conf(mol, 1)
    assert np.all(x0 == x1_new)

    with pytest.raises(ValueError, match="Bad Conformer Id"):
        _ = utils.get_romol_conf(mol, 2)

    with pytest.raises(ValueError, match="Bad Conformer Id"):
        _ = utils.set_romol_conf(mol, x0, 2)


def test_experimental_conversions_to_kj():
    rng = np.random.default_rng(2022)

    experimental_values = rng.random(10)

    # Assert that the temperature kwarg defaults to the same temperature that simulations default to.
    # TBD: Investigate changing DEFAULT_TEMP to 298.15 K
    np.testing.assert_array_equal(
        utils.convert_uM_to_kJ_per_mole(experimental_values),
        utils.convert_uM_to_kJ_per_mole(experimental_values, experiment_temp=DEFAULT_TEMP),
    )
    # Verify that uM to kJ and uIC50 to Kj is identical
    np.testing.assert_array_equal(
        utils.convert_uM_to_kJ_per_mole(experimental_values), utils.convert_uIC50_to_kJ_per_mole(experimental_values)
    )

    np.testing.assert_allclose(utils.convert_uM_to_kJ_per_mole(0.15), -39.192853)


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
    assert model_utils.get_strained_atoms(mol, ff) == [4, 10, 11]


def test_view_atom_mapping_3d():
    mol_a = Chem.AddHs(Chem.MolFromSmiles("c1ccc1"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1(N)ccc1"))

    AllChem.EmbedMolecule(mol_a)
    AllChem.EmbedMolecule(mol_b)

    # no core
    view = utils.view_atom_mapping_3d(mol_a, mol_b)
    assert isinstance(view, py3Dmol.view)

    view = utils.view_atom_mapping_3d(mol_a, mol_b, [])
    assert isinstance(view, py3Dmol.view)

    view = utils.view_atom_mapping_3d(mol_a, mol_b, np.array([]))
    assert isinstance(view, py3Dmol.view)

    # single core
    core = [[2, 0], [3, 2], [0, 3], [1, 4]]

    view = utils.view_atom_mapping_3d(mol_a, mol_b, [core])
    assert isinstance(view, py3Dmol.view)

    with pytest.raises(AssertionError, match="expect a list of cores"):
        utils.view_atom_mapping_3d(mol_a, mol_b, core)

    # multiple cores
    view = utils.view_atom_mapping_3d(mol_a, mol_b, [core, core])
    assert isinstance(view, py3Dmol.view)

    # multiple cores, ndarray input
    view = utils.view_atom_mapping_3d(mol_a, mol_b, np.array([core, core]))
    assert isinstance(view, py3Dmol.view)

    # multiple cores, different sizes
    cores = [
        [[2, 0], [3, 2], [0, 3]],
        [[2, 0], [3, 2], [0, 3], [1, 4]],
    ]

    view = utils.view_atom_mapping_3d(mol_a, mol_b, cores)
    assert isinstance(view, py3Dmol.view)

    # multiple cores, different sizes, ndarray input
    cores = [
        np.array([[2, 0], [3, 2], [0, 3]]),
        np.array([[2, 0], [3, 2], [0, 3], [1, 4]]),
    ]

    view = utils.view_atom_mapping_3d(mol_a, mol_b, cores)
    assert isinstance(view, py3Dmol.view)
