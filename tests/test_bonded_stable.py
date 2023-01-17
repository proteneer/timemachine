import numpy as np
import pytest
from common import GradientTest

import timemachine.potentials.generic as potentials


@pytest.mark.parametrize("n_particles", [64])
@pytest.mark.parametrize("n_angles", [25])
def test_harmonic_angle_stable(n_particles, n_angles):
    """Randomly connect triples of particles, then validate the resulting HarmonicAngleStable force"""
    np.random.seed(125)

    x = GradientTest().get_random_coords(n_particles, D=3)

    atom_idxs = np.arange(n_particles)
    params = np.random.rand(n_angles, 3).astype(np.float64)
    angle_idxs = []
    for _ in range(n_angles):
        angle_idxs.append(np.random.choice(atom_idxs, size=3, replace=False))
    angle_idxs = np.array(angle_idxs, dtype=np.int32) if n_angles else np.zeros((0, 3), dtype=np.int32)

    box = np.eye(3) * 100  # note: ignored

    # specific to harmonic angle force
    relative_tolerance_at_precision = {np.float32: 2e-5, np.float64: 1e-9}

    potential = potentials.HarmonicAngleStable(angle_idxs)

    for precision, rtol in relative_tolerance_at_precision.items():
        GradientTest().compare_forces_gpu_vs_reference(x, [params], box, potential, rtol, precision=precision)

        # test bitwise commutativity
        test_potential = potentials.HarmonicAngleStable(angle_idxs).to_gpu()
        test_potential_rev = potentials.HarmonicAngleStable(angle_idxs[:, ::-1]).to_gpu()

        test_potential_impl = test_potential.unbound_impl(precision)
        test_potential_rev_impl = test_potential_rev.unbound_impl(precision)

        test_du_dx, test_du_dp, test_u = test_potential_impl.execute_selective(x, params, box, 1, 1, 1)
        test_du_dx_rev, test_du_dp_rev, test_u_rev = test_potential_rev_impl.execute_selective(x, params, box, 1, 1, 1)

        np.testing.assert_array_equal(test_u, test_u_rev)
        np.testing.assert_array_equal(test_du_dx, test_du_dx_rev)
        np.testing.assert_array_equal(test_du_dp, test_du_dp_rev)
