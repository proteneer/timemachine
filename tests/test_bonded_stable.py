import numpy as np
import pytest
from common import GradientTest

from timemachine.potentials import HarmonicAngle, HarmonicAngleStable


def generate_system(n_particles, n_angles, seed):
    np.random.seed(seed)
    coords = GradientTest().get_random_coords(n_particles, D=3)
    params = np.random.rand(n_angles, 3).astype(np.float64)

    angle_idxs = []
    for _ in range(n_angles):
        angle_idxs.append(np.random.choice(n_particles, size=3, replace=False))
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
    potential = HarmonicAngleStable(angle_idxs)
    test_impl = potential.to_gpu(precision)
    GradientTest().compare_forces(coords, params, box, potential, test_impl, rtol)
    GradientTest().assert_differentiable_interface_consistency(coords, params, box, test_impl)


@pytest.mark.parametrize("n_particles", [64])
@pytest.mark.parametrize("n_angles", [25])
@pytest.mark.parametrize("precision", [np.float32, np.float64])
@pytest.mark.parametrize("seed", [2022])
def test_harmonic_angle_stable_bitwise_symmetric(n_particles, n_angles, precision, seed):
    "Test bitwise equality when angles are transformed like (i, j, k) -> (k, j, i)"

    angle_idxs, coords, params = generate_system(n_particles, n_angles, seed)

    test_potential_impl = HarmonicAngleStable(angle_idxs).to_gpu(precision).unbound_impl
    test_potential_rev_impl = HarmonicAngleStable(angle_idxs[:, ::-1]).to_gpu(precision).unbound_impl

    box = np.eye(3) * 100  # note: ignored
    test_du_dx, test_du_dp, test_u = test_potential_impl.execute(coords, params, box, 1, 1, 1)
    test_du_dx_rev, test_du_dp_rev, test_u_rev = test_potential_rev_impl.execute(coords, params, box, 1, 1, 1)

    np.testing.assert_array_equal(test_u, test_u_rev)
    np.testing.assert_array_equal(test_du_dx, test_du_dx_rev)
    np.testing.assert_array_equal(test_du_dp, test_du_dp_rev)


@pytest.mark.parametrize("n_particles", [64])
@pytest.mark.parametrize("n_angles", [25])
@pytest.mark.parametrize("precision,rtol", [(np.float32, 2e-5), (np.float64, 1e-9)])
@pytest.mark.parametrize("seed", [2022])
def test_harmonic_angle_stable_reduces_to_harmonic_angle(n_particles, n_angles, precision, rtol, seed):
    "Check that the stable functional form is equivalent to the standard one at eps = 0"

    angle_idxs, coords, params = generate_system(n_particles, n_angles, seed)
    params[:, 2] = 0.0  # param idx 2 is eps

    impl = HarmonicAngle(angle_idxs).to_gpu(precision).unbound_impl
    impl_stable = HarmonicAngleStable(angle_idxs).to_gpu(precision).unbound_impl

    box = np.eye(3) * 100  # note: ignored
    du_dx, du_dp, u = impl.execute(coords, params[:, :2], box, 1, 1, 1)
    du_dx_stable, du_dp_stable, u_stable = impl_stable.execute(coords, params, box, 1, 1, 1)

    np.testing.assert_allclose(u, u_stable, rtol=rtol)
    np.testing.assert_allclose(du_dx, du_dx_stable, rtol=rtol)
    np.testing.assert_allclose(du_dp, du_dp_stable[:, :2], rtol=rtol)


@pytest.mark.parametrize(
    "potential,params",
    [
        pytest.param(HarmonicAngle, [(1, 1)], marks=pytest.mark.xfail(reason="expect singularity", strict=True)),
        pytest.param(HarmonicAngleStable, [(1, 1, 1)]),
        pytest.param(
            HarmonicAngleStable,
            [(1, 1, 0)],
            marks=pytest.mark.xfail(reason="expect singularity when eps=0", strict=True),
        ),
    ],
)
@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_harmonic_angle_finite_force_with_vanishing_bond_length(potential, params, precision):
    "Check that forces do not blow up when a bond has length close to zero"

    angle_idxs = [(0, 1, 2)]
    coords = [(0, 0, 0), (1e-9, 0, 0), (0, 1, 0)]
    impl = potential(angle_idxs).to_gpu(precision).unbound_impl
    box = np.eye(3) * 100  # note: ignored
    du_dx, _, _ = impl.execute(coords, params, box, 1, 0, 0)
    print(du_dx)
    assert (np.abs(du_dx) < 1e7).all()
