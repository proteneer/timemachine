import numpy as np
import pytest
from common import GradientTest

import timemachine.potentials.generic as potentials


def generate_system(n_particles, n_angles, seed):
    np.random.seed(seed)
    coords = GradientTest().get_random_coords(n_particles, D=3)
    atom_idxs = np.arange(n_particles)
    params = np.random.rand(n_angles, 3).astype(np.float64)

    angle_idxs = []
    for _ in range(n_angles):
        angle_idxs.append(np.random.choice(atom_idxs, size=3, replace=False))
    angle_idxs = np.array(angle_idxs, dtype=np.int32) if n_angles else np.zeros((0, 3), dtype=np.int32)

    return angle_idxs, coords, params


@pytest.mark.parametrize("n_particles", [64])
@pytest.mark.parametrize("n_angles", [25])
@pytest.mark.parametrize("precision,rtol", [(np.float32, 2e-5), (np.float64, 1e-9)])
@pytest.mark.parametrize("seed", [2022])
def test_harmonic_angle_stable(n_particles, n_angles, precision, rtol, seed):
    """Validate HarmonicAngleStable reference on random triples of particles"""

    box = np.eye(3) * 100  # note: ignored
    angle_idxs, coords, params = generate_system(n_particles, n_angles, seed)
    potential = potentials.HarmonicAngleStable(angle_idxs)
    GradientTest().compare_forces_gpu_vs_reference(coords, [params], box, potential, rtol, precision=precision)


@pytest.mark.parametrize("n_particles", [64])
@pytest.mark.parametrize("n_angles", [25])
@pytest.mark.parametrize("precision", [np.float32, np.float64])
@pytest.mark.parametrize("seed", [2022])
def test_harmonic_angle_stable_bitwise_symmetric(n_particles, n_angles, precision, seed):
    "Test bitwise equality when angles are transformed like (i, j, k) -> (k, j, i)"

    angle_idxs, coords, params = generate_system(n_particles, n_angles, seed)

    test_potential = potentials.HarmonicAngleStable(angle_idxs).to_gpu()
    test_potential_rev = potentials.HarmonicAngleStable(angle_idxs[:, ::-1]).to_gpu()

    test_potential_impl = test_potential.unbound_impl(precision)
    test_potential_rev_impl = test_potential_rev.unbound_impl(precision)

    box = np.eye(3) * 100  # note: ignored
    test_du_dx, test_du_dp, test_u = test_potential_impl.execute_selective(coords, params, box, 1, 1, 1)
    test_du_dx_rev, test_du_dp_rev, test_u_rev = test_potential_rev_impl.execute_selective(coords, params, box, 1, 1, 1)

    np.testing.assert_array_equal(test_u, test_u_rev)
    np.testing.assert_array_equal(test_du_dx, test_du_dx_rev)
    np.testing.assert_array_equal(test_du_dp, test_du_dp_rev)
