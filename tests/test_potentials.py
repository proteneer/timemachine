import itertools
import pickle
import tempfile
from pathlib import Path

import jax
import numpy as np
import pytest
from common import GradientTest

from timemachine.lib import custom_ops
from timemachine.potentials import (
    BoundPotential,
    FanoutSummedPotential,
    HarmonicAngleStable,
    HarmonicBond,
    SummedPotential,
)

pytestmark = [pytest.mark.memcheck]


@pytest.fixture
def harmonic_bond():
    bond_idxs = np.array([[0, 1], [0, 2]], dtype=np.int32)
    params = np.ones(shape=(2, 2), dtype=np.float32)
    return HarmonicBond(bond_idxs).bind(params)


def execute_bound_impl(bp):
    coords = np.zeros(shape=(3, 3), dtype=np.float32)
    box = np.diag(np.ones(3))
    bp.execute(coords, box)


def test_bound_potential_keeps_referenced_potential_alive(harmonic_bond):
    bp = harmonic_bond.to_gpu(np.float32).bound_impl

    # segfaults if referenced potential has been deallocated prematurely
    execute_bound_impl(bp)


def test_bound_potential_get_potential(harmonic_bond):
    unbound_impl = harmonic_bond.potential.to_gpu(np.float32).unbound_impl
    bound_impl = custom_ops.BoundPotential(unbound_impl, harmonic_bond.params)
    assert unbound_impl is bound_impl.get_potential()


def test_bound_potential_empty_params():
    bond_idxs = np.empty((0, 2), dtype=np.int32)
    params = np.empty((0, 2))
    u_test = HarmonicBond(bond_idxs).bind(params).to_gpu(np.float32)
    x = np.empty((0, 3))
    box = np.eye(3)
    assert u_test(x, box) == 0.0

    u_test.bound_impl.set_params(np.empty((0, 2)))
    assert u_test(x, box) == 0.0


def test_bound_potential_set_params(harmonic_bond):
    x, box = np.ones((3, 3)), np.eye(3)

    new_params = np.random.default_rng(2023).random(size=(2, 2), dtype=np.float32)
    u_ref = harmonic_bond.potential.to_gpu(np.float32).bind(new_params)

    # before updating u_test with new parameters, should disagree with reference
    u_test = harmonic_bond.to_gpu(np.float32)
    assert not np.array_equal(u_test(x, box), u_ref(x, box))

    u_test.bound_impl.set_params(new_params)
    np.testing.assert_array_equal(u_test(x, box), u_ref(x, box))

    # should raise an exception if new parameters size != buffer size
    with pytest.raises(RuntimeError, match="2 != 4"):
        invalid_params = np.ones(shape=(1, 2), dtype=np.float32)
        u_test.bound_impl.set_params(invalid_params)


def verify_potential_validation(potential):
    with pytest.raises(RuntimeError, match="coords dimensions must be 2"):
        potential(np.zeros(1), np.ones((3, 3)))

    with pytest.raises(RuntimeError, match="coords must have a shape that is 3 dimensional"):
        potential(np.zeros((1, 4)), np.ones((3, 3)))

    with pytest.raises(RuntimeError, match="box must be 3x3"):
        potential(np.zeros((1, 3)), np.ones(3))

    with pytest.raises(RuntimeError, match="box must be 3x3"):
        potential(np.zeros((1, 3)), np.ones((2, 2)))

    with pytest.raises(RuntimeError, match="box must be ortholinear"):
        potential(np.zeros((1, 3)), np.ones((3, 3)))

    with pytest.raises(RuntimeError, match="box must have positive values along diagonal"):
        potential(np.zeros((1, 3)), np.eye(3) * 0.0)


def test_bound_potential_execute_validation(harmonic_bond):
    bound_impl = harmonic_bond.to_gpu(np.float32).bound_impl
    verify_potential_validation(bound_impl.execute)

    execute_bound_impl(bound_impl)


def test_unbound_potential_execute_validation(harmonic_bond):
    unbound_impl = harmonic_bond.potential.to_gpu(np.float32).unbound_impl

    def func(coords, box):
        return unbound_impl.execute(coords, harmonic_bond.params, box)

    verify_potential_validation(func)

    unbound_impl.execute(np.zeros((3, 3)), harmonic_bond.params, np.eye(3))


def test_summed_potential_raises_on_inconsistent_lengths(harmonic_bond):
    with pytest.raises(ValueError) as excinfo:
        SummedPotential([harmonic_bond], [])

    assert str(excinfo.value) == "number of potentials != number of parameter arrays"


def test_summed_potential_keeps_referenced_potentials_alive(harmonic_bond):
    sp = SummedPotential([harmonic_bond.potential], [harmonic_bond.params]).bind(harmonic_bond.params)

    # segfaults if referenced potentials have been deallocated prematurely
    execute_bound_impl(sp.to_gpu(np.float32).bound_impl)


def test_summed_potential_get_potentials(harmonic_bond):
    impls = [harmonic_bond.potential.to_gpu(np.float32).unbound_impl for _ in range(2)]
    params_sizes = [harmonic_bond.params.size] * 2
    summed_impl = custom_ops.SummedPotential(impls, params_sizes)
    assert set(id(p) for p in summed_impl.get_potentials()) == set(id(p) for p in impls)


def test_summed_potential_invalid_parameters_size(harmonic_bond):
    sp = SummedPotential([harmonic_bond.potential], [harmonic_bond.params])

    with pytest.raises(RuntimeError) as e:
        execute_bound_impl(sp.bind(np.empty(0)).to_gpu(np.float32).bound_impl)
    assert f"SummedPotential::execute_device(): expected {harmonic_bond.params.size} parameters, got 0" in str(e)

    with pytest.raises(RuntimeError) as e:
        execute_bound_impl(sp.bind(np.ones(harmonic_bond.params.size + 1)).to_gpu(np.float32).bound_impl)
    assert (
        f"SummedPotential::execute_device(): expected {harmonic_bond.params.size} parameters, got {harmonic_bond.params.size + 1}"
        in str(e)
    )

    # test reference potential
    x, box = np.ones((3, 3)), np.eye(3)

    bp = sp.bind(np.empty(0))
    with pytest.raises(AssertionError):
        _ = bp(x, box)

    bp = sp.bind(np.ones(harmonic_bond.params.size + 1))
    with pytest.raises(AssertionError):
        _ = bp(x, box)

    # should assert flattened params
    bp = sp.bind(harmonic_bond.params)
    with pytest.raises(AssertionError):
        _ = bp(x, box)


def test_summed_potential_nested(harmonic_bond):
    nested_sp = SummedPotential([harmonic_bond.potential], [harmonic_bond.params])
    nested_sp_params = harmonic_bond.params.flatten()
    sp = SummedPotential([nested_sp, harmonic_bond.potential], [nested_sp_params, harmonic_bond.params])
    sp_params = np.concatenate([nested_sp_params, harmonic_bond.params.flatten()])
    bp = sp.bind(sp_params)
    execute_bound_impl(bp.to_gpu(np.float32).bound_impl)

    # test reference potential
    x, box = np.ones((3, 3)), np.eye(3)
    _ = bp(x, box)

    # another level of nesting is fine, too
    sp_prime = SummedPotential(
        [sp, nested_sp, harmonic_bond.potential], [sp_params, nested_sp_params, harmonic_bond.params]
    )
    sp_prime_params = np.concatenate([sp_params, nested_sp_params, harmonic_bond.params.flatten()])
    bp_prime = sp_prime.bind(sp_prime_params)
    _ = bp_prime(x, box)


def reference_execute_over_batch(unbound, coords, boxes, params):
    coord_batches = coords.shape[0]
    param_batches = params.shape[0]
    N = coords.shape[1]
    D = coords.shape[2]
    du_dx = np.empty((coord_batches, param_batches, N, D))
    du_dp = np.empty((coord_batches, param_batches, *params.shape[1:]))
    u = np.empty((coord_batches, param_batches))
    for i in range(coord_batches):
        for j in range(param_batches):
            ref_du_dx, ref_du_dp, ref_u = unbound.execute(coords[i], params[j], boxes[i])
            du_dx[i][j] = ref_du_dx
            du_dp[i][j] = ref_du_dp
            u[i][j] = ref_u
    return du_dx, du_dp, u


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_unbound_impl_execute_batch(harmonic_bond, precision):
    np.random.seed(2022)

    N = 5

    coords = np.random.random((N, 3))
    perturbed_coords = coords + np.random.random(coords.shape)

    num_coord_batches = 5
    num_param_batches = 3

    box = np.diag(np.ones(3))
    coords_batch = np.stack([coords, perturbed_coords] * num_coord_batches)
    boxes_batch = np.stack([box] * 2 * num_coord_batches)

    params = harmonic_bond.params
    random_params = np.random.random(params.shape)

    params_batch = np.stack([params, random_params] * num_param_batches)

    unbound_impl = harmonic_bond.potential.to_gpu(precision).unbound_impl

    ref_du_dx, ref_du_dp, ref_u = reference_execute_over_batch(unbound_impl, coords_batch, boxes_batch, params_batch)

    # Verify that number of boxes and coords match
    with pytest.raises(RuntimeError) as e:
        _ = unbound_impl.execute_batch(
            coords_batch,
            params_batch,
            boxes_batch[:num_coord_batches],
            True,
            True,
            True,
        )
    assert str(e.value) == "number of batches of coords and boxes don't match"

    # Verify that coords have 3 dimensions
    with pytest.raises(RuntimeError) as e:
        _ = unbound_impl.execute_batch(
            coords,
            params_batch,
            box,
            True,
            True,
            True,
        )
    assert str(e.value) == "coords and boxes must have 3 dimensions"

    # Verify that params must have at least two dimensions
    with pytest.raises(RuntimeError) as e:
        _ = unbound_impl.execute_batch(
            coords_batch,
            np.ones(3),
            boxes_batch,
            True,
            True,
            True,
        )
    assert str(e.value) == "parameters must have at least 2 dimensions"

    shape_prefix = (len(coords_batch), len(params_batch))

    for combo in itertools.product([False, True], repeat=3):
        compute_du_dx, compute_du_dp, compute_u = combo
        batch_du_dx, batch_du_dp, batch_u = unbound_impl.execute_batch(
            coords_batch,
            params_batch,
            boxes_batch,
            compute_du_dx,
            compute_du_dp,
            compute_u,
        )
        if compute_du_dx:
            assert batch_du_dx.shape == (*shape_prefix, N, 3)
            np.testing.assert_array_equal(batch_du_dx, ref_du_dx)
        else:
            assert batch_du_dx is None

        if compute_du_dp:
            assert batch_du_dp.shape == (*shape_prefix, *harmonic_bond.params.shape)
            np.testing.assert_array_equal(batch_du_dp, ref_du_dp)
        else:
            assert batch_du_dp is None

        if compute_u:
            assert batch_u.shape == (*shape_prefix,)
            np.testing.assert_array_equal(batch_u, ref_u)
        else:
            assert batch_u is None


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_bound_impl_execute_batch(harmonic_bond, precision):
    np.random.seed(2022)

    N = 5

    coords = np.random.random((N, 3))
    perturbed_coords = coords + np.random.random(coords.shape)

    num_coord_batches = 5

    box = np.diag(np.ones(3))
    coords_batch = np.stack([coords, perturbed_coords] * num_coord_batches)
    boxes_batch = np.stack([box] * 2 * num_coord_batches)

    params = harmonic_bond.params

    unbound_gpu = harmonic_bond.potential.to_gpu(precision)
    bound_impl = unbound_gpu.bind(params).bound_impl

    ref_du_dx, _, ref_u = reference_execute_over_batch(
        unbound_gpu.unbound_impl, coords_batch, boxes_batch, np.array([params])
    )
    # Remove the dimension for parameters that don't matter
    ref_du_dx = ref_du_dx.squeeze()
    ref_u = ref_u.squeeze()

    # Verify that number of boxes and coords match
    with pytest.raises(RuntimeError) as e:
        _ = bound_impl.execute_batch(
            coords_batch,
            boxes_batch[: num_coord_batches - 1],
            True,
            True,
        )
    assert str(e.value) == "number of batches of coords and boxes don't match"

    # Verify that coords have 3 dimensions
    with pytest.raises(RuntimeError) as e:
        _ = bound_impl.execute_batch(
            coords,
            box,
            True,
            True,
        )
    assert str(e.value) == "coords and boxes must have 3 dimensions"

    for combo in itertools.product([False, True], repeat=2):
        compute_du_dx, compute_u = combo
        batch_du_dx, batch_u = bound_impl.execute_batch(
            coords_batch,
            boxes_batch,
            compute_du_dx,
            compute_u,
        )
        if compute_du_dx:
            assert batch_du_dx.shape == (len(coords_batch), N, 3)
            np.testing.assert_array_equal(batch_du_dx, ref_du_dx)
        else:
            assert batch_du_dx is None

        if compute_u:
            assert batch_u.shape == (len(coords_batch),)
            np.testing.assert_array_equal(batch_u, ref_u)
        else:
            assert batch_u is None


@pytest.fixture
def harmonic_bond_test_system():
    np.random.seed(2022)

    num_atoms = 10
    num_bonds = 10

    coords = np.random.uniform(0, 1, size=(num_atoms, 3)).astype(np.float32)

    def random_bond_idxs():
        return np.array(
            [np.random.choice(num_atoms, size=(2,), replace=False) for _ in range(num_bonds)], dtype=np.int32
        )

    harmonic_bond_1 = HarmonicBond(random_bond_idxs())
    harmonic_bond_2 = HarmonicBond(random_bond_idxs())

    params_1 = np.random.uniform(0, 1, size=(num_bonds, 2))
    params_2 = np.random.uniform(0, 1, size=(num_bonds, 2))

    return harmonic_bond_1, harmonic_bond_2, params_1, params_2, coords


@pytest.mark.parametrize("num_potentials", [1, 2, 5])
@pytest.mark.parametrize("parallel", [False, True])
def test_summed_potential(parallel, num_potentials, harmonic_bond_test_system):
    """Assert SummedPotential is consistent on a set of harmonic bond potentials"""

    harmonic_bond, _, params, _, coords = harmonic_bond_test_system

    box = 3.0 * np.eye(3)
    params_list = [params] * num_potentials
    potential = SummedPotential([harmonic_bond] * num_potentials, params_list, parallel)

    flat_params = np.concatenate([p.reshape(-1) for p in params_list])

    for rtol, precision in [(1e-6, np.float32), (1e-10, np.float64)]:
        GradientTest().compare_forces(coords, flat_params, box, potential, potential.to_gpu(precision), rtol)


@pytest.mark.parametrize("parallel", [False, True])
def test_fanout_summed_potential_consistency(parallel, harmonic_bond_test_system):
    """Assert FanoutSummedPotential consistent with SummedPotential on
    a harmonic bond instance"""

    harmonic_bond_1, harmonic_bond_2, params, _, coords = harmonic_bond_test_system

    summed_potential = SummedPotential([harmonic_bond_1, harmonic_bond_2], [params, params])

    fanout_summed_potential = FanoutSummedPotential([harmonic_bond_1, harmonic_bond_2], parallel)

    box = 3.0 * np.eye(3)

    du_dx_ref, du_dps_ref, u_ref = summed_potential.to_gpu(np.float32).unbound_impl.execute(
        coords, [params, params], box
    )

    du_dx_test, du_dp_test, u_test = fanout_summed_potential.to_gpu(np.float32).unbound_impl.execute(
        coords, params, box
    )

    np.testing.assert_array_equal(du_dx_ref, du_dx_test)
    np.testing.assert_allclose(np.sum(du_dps_ref, axis=0), du_dp_test, rtol=1e-8, atol=1e-8)
    assert u_ref == u_test


def test_potential_jax_differentiable(harmonic_bond):
    potential = harmonic_bond.potential
    params = harmonic_bond.params
    coords = np.zeros(shape=(3, 3), dtype=np.float32)
    box = np.diag(np.ones(3))
    du_dx, du_dp = jax.grad(potential, argnums=(0, 1))(coords, params, box)


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_bound_and_unbound_execute_match(harmonic_bond_test_system, precision):
    """Verify that whether using the bound or unbound implementation of a potential the forces and energies computed are bitwise identical."""
    harmonic_bond_1, _, params, _, coords = harmonic_bond_test_system

    gpu_bond = harmonic_bond_1.to_gpu(precision)

    gpu_bound_bond = gpu_bond.bind(params)

    box = 3.0 * np.eye(3)

    unbound_impl = gpu_bond.unbound_impl
    bound_impl = gpu_bound_bond.bound_impl
    for combo in itertools.product([False, True], repeat=2):
        compute_du_dx, compute_u = combo

        bound_du_dx, bound_u = bound_impl.execute(coords, box, compute_u=compute_u, compute_du_dx=compute_du_dx)

        unbound_du_dx, _, unbound_u = unbound_impl.execute(
            coords, params, box, compute_u=compute_u, compute_du_dp=False, compute_du_dx=compute_du_dx
        )
        np.testing.assert_array_equal(bound_du_dx, unbound_du_dx)
        np.testing.assert_array_equal(bound_u, unbound_u)


def test_execute_batch_sparse_validation(harmonic_bond: BoundPotential[HarmonicBond]):
    unbound_impl = harmonic_bond.potential.to_gpu(np.float32).unbound_impl

    # Should verify that number of boxes and coords match
    with pytest.raises(RuntimeError) as e:
        _ = unbound_impl.execute_batch_sparse(
            np.zeros((5, 4, 3)),
            np.zeros((4, 3, 2)),
            np.zeros((6, 3, 3)),  # inconsistent length
            np.zeros(3).astype(np.uint32),
            np.zeros(3).astype(np.uint32),
            True,
            True,
            True,
        )
    assert str(e.value) == "number of coord arrays and boxes don't match"

    # Should verify that coords and boxes have 3 dimensions
    with pytest.raises(RuntimeError) as e:
        _ = unbound_impl.execute_batch_sparse(
            np.zeros((5, 4, 3)),
            np.zeros((4, 3, 2)),
            np.zeros((5, 3)),  # missing dimension
            np.zeros(3).astype(np.uint32),
            np.zeros(3).astype(np.uint32),
            True,
            True,
            True,
        )
    assert str(e.value) == "coords and boxes must have 3 dimensions"

    with pytest.raises(RuntimeError) as e:
        _ = unbound_impl.execute_batch_sparse(
            np.zeros((5, 4)),  # missing dimension
            np.zeros((4, 3, 2)),
            np.zeros((5, 3, 3)),
            np.zeros(3).astype(np.uint32),
            np.zeros(3).astype(np.uint32),
            True,
            True,
            True,
        )
    assert str(e.value) == "coords and boxes must have 3 dimensions"

    # Should verify that params have at least two dimensions
    with pytest.raises(RuntimeError) as e:
        _ = unbound_impl.execute_batch_sparse(
            np.zeros((5, 4, 3)),
            np.zeros(4),  # 1-d
            np.zeros((5, 3, 3)),
            np.zeros(3).astype(np.uint32),
            np.zeros(3).astype(np.uint32),
            True,
            True,
            True,
        )
    assert str(e.value) == "parameters must have at least 2 dimensions"

    # Should verify that coords_batch_idxs and params_batch_idxs are 1-d
    with pytest.raises(RuntimeError) as e:
        _ = unbound_impl.execute_batch_sparse(
            np.zeros((5, 4, 3)),
            np.zeros((4, 3, 2)),
            np.zeros((5, 3, 3)),
            np.zeros((3, 1)).astype(np.uint32),  # 2-d
            np.zeros(3).astype(np.uint32),
            True,
            True,
            True,
        )
    assert str(e.value) == "coords_batch_idxs and params_batch_idxs must be one-dimensional arrays"

    with pytest.raises(RuntimeError) as e:
        _ = unbound_impl.execute_batch_sparse(
            np.zeros((5, 4, 3)),
            np.zeros((4, 3, 2)),
            np.zeros((5, 3, 3)),
            np.zeros(3).astype(np.uint32),
            np.array(0).astype(np.uint32),  # 0-d
            True,
            True,
            True,
        )
    assert str(e.value) == "coords_batch_idxs and params_batch_idxs must be one-dimensional arrays"

    # Should verify that coords_batch_idxs and params_batch_idxs have the same length
    with pytest.raises(RuntimeError) as e:
        _ = unbound_impl.execute_batch_sparse(
            np.zeros((5, 4, 3)),
            np.zeros((4, 3, 2)),
            np.zeros((5, 3, 3)),
            np.zeros(3).astype(np.uint32),
            np.zeros(4).astype(np.uint32),  # inconsistent length
            True,
            True,
            True,
        )
    assert str(e.value) == "coords_batch_idxs and params_batch_idxs must have the same length"

    # Should verify that coords_batch_idxs and params_batch_idxs are in bounds
    with pytest.raises(RuntimeError) as e:
        _ = unbound_impl.execute_batch_sparse(
            np.zeros((5, 4, 3)),
            np.zeros((4, 3, 2)),
            np.zeros((5, 3, 3)),
            np.array([5]).astype(np.uint32),  # out of bounds
            np.array([0]).astype(np.uint32),
            True,
            True,
            True,
        )
    assert str(e.value) == "coords_batch_idxs contains an index that is out of bounds"

    with pytest.raises(RuntimeError) as e:
        _ = unbound_impl.execute_batch_sparse(
            np.zeros((5, 4, 3)),
            np.zeros((4, 3, 2)),
            np.zeros((5, 3, 3)),
            np.array([0]).astype(np.uint32),
            np.array([4]).astype(np.uint32),  # out of bounds
            True,
            True,
            True,
        )
    assert str(e.value) == "params_batch_idxs contains an index that is out of bounds"


@pytest.mark.parametrize("precision", [np.float32, np.float64])
@pytest.mark.parametrize("coords_size", [1, 5])
@pytest.mark.parametrize("params_size", [1, 5])
@pytest.mark.parametrize("batch_size", [1, 5, 10])
@pytest.mark.parametrize("seed", [2024, 2025])
def test_execute_batch_sparse(
    harmonic_bond: BoundPotential[HarmonicBond], precision, coords_size, params_size, batch_size, seed
):
    rng = np.random.default_rng(seed)

    n_atoms = 5
    coords = rng.normal(0, 1, (coords_size, n_atoms, 3))

    params = rng.uniform(size=(params_size, *harmonic_bond.params.shape))
    boxes = np.eye(3) * rng.uniform(size=(len(coords), 3))[:, :, np.newaxis]

    coords_batch_idxs = rng.choice(coords_size, batch_size).astype(np.uint32)
    params_batch_idxs = rng.choice(params_size, batch_size).astype(np.uint32)

    unbound_impl = harmonic_bond.potential.to_gpu(precision).unbound_impl

    def run_reference(flags):
        results = [
            unbound_impl.execute(coords[coords_idx], params[params_idx], boxes[coords_idx], *flags)
            for coords_idx, params_idx in zip(coords_batch_idxs, params_batch_idxs)
        ]
        du_dx, du_dp, u = zip(*results)
        return np.array(du_dx), np.array(du_dp), np.array(u)

    for flags in itertools.product([False, True], repeat=3):
        compute_du_dx, compute_du_dp, compute_u = flags
        ref_du_dx, ref_du_dp, ref_u = run_reference(flags)
        du_dx, du_dp, u = unbound_impl.execute_batch_sparse(
            coords,
            params,
            boxes.astype(np.float64),
            coords_batch_idxs,
            params_batch_idxs,
            *flags,
        )
        if compute_du_dx:
            assert du_dx.shape == (batch_size, n_atoms, 3)
            np.testing.assert_array_equal(du_dx, ref_du_dx)
        else:
            assert du_dx is None

        if compute_du_dp:
            assert du_dp.shape == (batch_size, *harmonic_bond.params.shape)
            np.testing.assert_array_equal(du_dp, ref_du_dp)
        else:
            assert du_dp is None

        if compute_u:
            assert u.shape == (batch_size,)
            np.testing.assert_array_equal(u, ref_u)
        else:
            assert u is None


def test_pickle_deprecated_harmonic_angle_stable():
    # Test deprecation behavior of the old HarmonicAngleStable class.
    # 1) This class is no longer serializable, users should convert instances to the HarmonicAngle class instead.
    # 2) Unpickling will raise a DeprecationWarning
    # 3) Initialization will raise a DeprecationWarning
    with open(Path(__file__).parent / "data" / "old_harmonic_angle_stable.pkl", "rb") as ifs:
        with pytest.warns(DeprecationWarning):
            initial_state, _ = pickle.load(ifs)
            assert isinstance(initial_state.potentials[1].potential, HarmonicAngleStable)

    idxs = np.array([[0, 1, 2]], dtype=np.int32)

    with pytest.warns(DeprecationWarning):
        potential = HarmonicAngleStable(idxs)
        with tempfile.NamedTemporaryFile() as tmp_file:
            with pytest.raises(NotImplementedError):
                pickle.dump(potential, tmp_file)
