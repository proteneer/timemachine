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

    rotation = Rotation.from_quat([convert_quaternion_for_scipy(quat) for quat in quaternions])
    for i, coords in enumerate(all_coords):
        ref_rotated_coords = rotation.apply(coords)
        for j, ref_coords in enumerate(ref_rotated_coords):
            np.testing.assert_allclose(
                rotated_coords[i][j],
                ref_coords,
                rtol=rtol,
                atol=atol,
                err_msg=f"Coords {i} with Rotation {j} have a mismatch",
            )
