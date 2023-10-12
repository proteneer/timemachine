import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from timemachine.lib import custom_ops


def hamiliton_product(q1: NDArray, q2: NDArray) -> NDArray:
    """
    Compute the product of two quaternions, called a hamilton product.

    References
    ----------
    https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    output = np.empty(4)
    output[0] = (w1 * w2) - (x1 * x2) - (y1 * y2) - (z1 * z2)
    output[1] = (w1 * x2) + (x1 * w2) + (y1 * z2) - (z1 * y2)
    output[2] = (w1 * y2) - (x1 * z2) + (y1 * w2) + (z1 * x2)
    output[3] = (w1 * z2) + (x1 * y2) - (y1 * x2) + (z1 * w2)
    return output


def rotate_vector_by_quaternions(coords: NDArray, quaternion: NDArray) -> NDArray:
    """
    Reference implementation of rotating coordinates using a quaternion

    References
    ----------
    https://math.stackexchange.com/a/535223
    """
    assert coords.shape == (3,)
    assert quaternion.shape[0] == 4
    expanded_coords = np.concatenate([np.zeros(1), coords])
    quaternion_conjugate = quaternion.copy()
    # Compute quaternion conjugate by negating all but w
    quaternion_conjugate[1:] *= -1
    ham_coords = hamiliton_product(quaternion, expanded_coords)
    return hamiliton_product(ham_coords, quaternion_conjugate)[1:]  # Truncate off w


def convert_quaternion_for_scipy(quat: NDArray) -> NDArray:
    """Scipy has the convention of (x, y, z, w) which is different than the wikipedia definition, swap ordering to verify using scipy.

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
    """
    return np.append(quat[1:], [quat[0]])


@pytest.mark.nogpu
@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize("size", [1000])
def test_reference_rotation_by_quaternion(seed, size):
    rng = np.random.default_rng(seed)

    all_coords = rng.normal(size=(size, 3)) * 100.0

    quaternions = rng.normal(size=(size, 4))
    # Normalize the quaternions to unit quaternions, representing a rotations
    norms = np.linalg.norm(quaternions, axis=-1)
    quaternions = quaternions / np.expand_dims(norms, -1)
    for coords, quaternion in zip(all_coords, quaternions):

        rotation = Rotation.from_quat(convert_quaternion_for_scipy(quaternion))

        ref_coords = rotation.apply(coords)

        test_coords = rotate_vector_by_quaternions(coords, quaternion)

        np.testing.assert_allclose(ref_coords, test_coords)


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

    rotated_coords = rotate_function(all_coords, quaternions)

    norms = np.linalg.norm(quaternions, axis=-1)
    # Normalize the quaternions to a unit quaternion after passing to C++, as should be automatically normalized
    quaternions = quaternions / np.expand_dims(norms, -1)
    # Generates rotations for all individual sets of coordinates
    assert rotated_coords.shape == (n_coords, n_rotations, 3)
    for i, coords in enumerate(all_coords):
        for j, quat in enumerate(quaternions):
            test_coords = rotate_vector_by_quaternions(coords, quat)

            np.testing.assert_allclose(
                rotated_coords[i][j],
                test_coords,
                rtol=rtol,
                atol=atol,
                err_msg=f"Coords {i} with Rotation {j} have a mismatch",
            )
