import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from timemachine.lib import custom_ops


def convert_quaternion_for_scipy(quat: NDArray) -> NDArray:
    """Scipy has the convention of (x, y, z, w) which is different than the wikipedia definition, swap ordering to verify using scipy.

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
    """
    return np.append(quat[1:], [quat[0]])


@pytest.mark.memcheck
@pytest.mark.parametrize("seed", [2023])
@pytest.mark.parametrize("precision,atol,rtol", [(np.float64, 1e-8, 1e-8), (np.float32, 2e-5, 1e-5)])
@pytest.mark.parametrize("n_rotations", [2, 33, 100])
@pytest.mark.parametrize("n_coords", [8, 33, 65])
def test_cuda_rotation_by_quaternion(seed, precision, atol, rtol, n_rotations, n_coords):
    rng = np.random.default_rng(seed)

    all_coords = rng.normal(size=(n_coords, 3)) * 100.0

    quaternions = rng.normal(size=(n_rotations, 4))
    rotate_function = custom_ops.rotate_coords_f32
    if precision == np.float64:
        rotate_function = custom_ops.rotate_coords_f64

    # Generates rotations for all individual sets of coordinates
    rotated_coords = rotate_function(all_coords, quaternions)
    assert rotated_coords.shape == (n_coords, n_rotations, 3)

    inv_quaternions = quaternions.copy()
    inv_quaternions[:, 1:] *= -1

    rotation = Rotation.from_quat([convert_quaternion_for_scipy(quat) for quat in quaternions])
    for i, coords in enumerate(all_coords):
        ref_rotated_coords = rotation.apply(coords)

        inverted_coords = rotate_function(ref_rotated_coords, inv_quaternions)

        for j, ref_coords in enumerate(ref_rotated_coords):
            np.testing.assert_allclose(
                rotated_coords[i][j],
                ref_coords,
                rtol=rtol,
                atol=atol,
                err_msg=f"Coords {i} with Rotation {j} have a mismatch",
            )
            np.testing.assert_allclose(
                inverted_coords[j][j],
                coords,
                rtol=rtol,
                atol=atol,
                err_msg=f"Coords {j} didn't get back to original value",
            )


@pytest.mark.memcheck
@pytest.mark.parametrize("seed", [2023])
@pytest.mark.parametrize("precision,atol,rtol", [(np.float64, 1e-8, 1e-8), (np.float32, 2e-5, 1e-5)])
@pytest.mark.parametrize("n_moves", [2, 33, 100])
@pytest.mark.parametrize("n_coords", [8, 33, 65])
def test_cuda_rotation_and_translation_by_quaternion(seed, precision, atol, rtol, n_moves, n_coords):
    rng = np.random.default_rng(seed)

    mol_coords = rng.normal(size=(n_coords, 3)) * 10.0
    box = np.eye(3) * 100.0

    quaternions = rng.normal(size=(n_moves, 4))
    translations = rng.uniform(size=(n_moves, 3))
    rotate_function = custom_ops.rotate_and_translate_mol_f32
    if precision == np.float64:
        rotate_function = custom_ops.rotate_and_translate_mol_f64

    for quat, translation in zip(quaternions, translations):
        rotated_translated_mol = rotate_function(mol_coords, box, quat, translation)

        translation_shift = np.diag(box) * translation
        assert rotated_translated_mol.shape == mol_coords.shape
        rotation = Rotation.from_quat(convert_quaternion_for_scipy(quat))

        ref_rotated_translated_mol = rotation.apply(mol_coords) + translation_shift

        np.testing.assert_allclose(
            rotated_translated_mol,
            ref_rotated_translated_mol,
            rtol=rtol,
            atol=atol,
        )
