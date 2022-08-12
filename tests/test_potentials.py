import jax

jax.config.update("jax_enable_x64", True)

import itertools

import numpy as np
import pytest
from common import GradientTest

from timemachine.lib import custom_ops
from timemachine.lib.potentials import FanoutSummedPotential, HarmonicBond, SummedPotential
from timemachine.potentials import generic

pytestmark = [pytest.mark.memcheck]


@pytest.fixture
def harmonic_bond():
    bond_idxs = np.array([[0, 1], [0, 2]], dtype=np.int32)
    params = np.ones(shape=(2, 2), dtype=np.float32)
    return HarmonicBond(bond_idxs).bind(params)


def execute_bound_impl(bp):
    coords = np.zeros(shape=(3, 3), dtype=np.float32)
    box = np.diag(np.ones(3))
    lam = 1
    bp.execute(coords, box, lam)


def test_bound_potential_keeps_referenced_potential_alive(harmonic_bond):
    bp = harmonic_bond.bound_impl(np.float32)

    # segfaults if referenced potential has been deallocated prematurely
    execute_bound_impl(bp)


def test_bound_potential_get_potential(harmonic_bond):
    unbound_impl = harmonic_bond.unbound_impl(np.float32)
    bound_impl = custom_ops.BoundPotential(unbound_impl, harmonic_bond.params)
    assert unbound_impl is bound_impl.get_potential()


def test_summed_potential_raises_on_inconsistent_lengths(harmonic_bond):
    with pytest.raises(ValueError) as excinfo:
        SummedPotential([harmonic_bond], [])

    assert str(excinfo.value) == "number of potentials != number of parameter arrays"


def test_summed_potential_keeps_referenced_potentials_alive(harmonic_bond):
    sp = SummedPotential([harmonic_bond], [harmonic_bond.params]).bind(harmonic_bond.params)

    # segfaults if referenced potentials have been deallocated prematurely
    execute_bound_impl(sp.bound_impl(np.float32))


def test_summed_potential_get_potentials(harmonic_bond):
    impls = [harmonic_bond.unbound_impl(np.float32) for _ in range(2)]
    params_sizes = [harmonic_bond.params.size] * 2
    summed_impl = custom_ops.SummedPotential(impls, params_sizes)
    assert set(id(p) for p in summed_impl.get_potentials()) == set(id(p) for p in impls)


def test_summed_potential_invalid_parameters_size(harmonic_bond):
    sp = SummedPotential([harmonic_bond], [harmonic_bond.params])

    with pytest.raises(RuntimeError) as e:
        execute_bound_impl(sp.bind(np.empty(0)).bound_impl(np.float32))
    assert f"SummedPotential::execute_device(): expected {harmonic_bond.params.size} parameters, got 0" in str(e)

    with pytest.raises(RuntimeError) as e:
        execute_bound_impl(sp.bind(np.ones(harmonic_bond.params.size + 1)).bound_impl(np.float32))
    assert (
        f"SummedPotential::execute_device(): expected {harmonic_bond.params.size} parameters, got {harmonic_bond.params.size + 1}"
        in str(e)
    )


def reference_execute_over_batch(unbound, coords, boxes, params, lambdas):
    coord_batches = coords.shape[0]
    param_batches = params.shape[0]
    lambda_batches = lambdas.size
    N = coords.shape[1]
    D = coords.shape[2]
    du_dx = np.empty((coord_batches, param_batches, lambda_batches, N, D))
    du_dp = np.empty((coord_batches, param_batches, lambda_batches, *params.shape[1:]))
    du_dl = np.empty((coord_batches, param_batches, lambda_batches))
    u = np.empty((coord_batches, param_batches, lambda_batches))
    for i in range(coord_batches):
        for j in range(param_batches):
            for k in range(lambda_batches):
                ref_du_dx, ref_du_dp, ref_du_dl, ref_u = unbound.execute(coords[i], params[j], boxes[i], lambdas[k])
                du_dx[i][j][k] = ref_du_dx
                du_dp[i][j][k] = ref_du_dp
                du_dl[i][j][k] = ref_du_dl
                u[i][j][k] = ref_u
    return du_dx, du_dp, du_dl, u


def test_execute_selective_batch(harmonic_bond):
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

    lambdas = np.array([0.0, 1.0])

    unbound_impl = harmonic_bond.unbound_impl(np.float32)

    ref_du_dx, ref_du_dp, ref_du_dl, ref_u = reference_execute_over_batch(
        unbound_impl, coords_batch, boxes_batch, params_batch, lambdas
    )

    # Verify that number of boxes and coords match
    with pytest.raises(RuntimeError) as e:
        _ = unbound_impl.execute_selective_batch(
            coords_batch,
            params_batch,
            boxes_batch[:num_coord_batches],
            lambdas,
            True,
            True,
            True,
            True,
        )
    assert str(e.value) == "number of batches of coords and boxes don't match"

    # Verify that coords have 3 dimensions
    with pytest.raises(RuntimeError) as e:
        _ = unbound_impl.execute_selective_batch(
            coords,
            params_batch,
            box,
            lambdas,
            True,
            True,
            True,
            True,
        )
    assert str(e.value) == "coords and boxes must have 3 dimensions"

    # Verify that params must have at least two dimensions
    with pytest.raises(RuntimeError) as e:
        _ = unbound_impl.execute_selective_batch(
            coords_batch,
            np.ones(3),
            boxes_batch,
            lambdas,
            True,
            True,
            True,
            True,
        )
    assert str(e.value) == "parameters must have at least 2 dimensions"

    shape_prefix = (len(coords_batch), len(params_batch), len(lambdas))

    for combo in itertools.product([False, True], repeat=4):
        compute_du_dx, compute_du_dp, compute_du_dl, compute_u = combo
        batch_du_dx, batch_du_dp, batch_du_dl, batch_u = unbound_impl.execute_selective_batch(
            coords_batch,
            params_batch,
            boxes_batch,
            lambdas,
            compute_du_dx,
            compute_du_dp,
            compute_du_dl,
            compute_u,
        )
        if compute_du_dx:
            assert batch_du_dx.shape == (*shape_prefix, N, 3)
        else:
            assert batch_du_dx is None
        if compute_du_dp:
            assert batch_du_dp.shape == (*shape_prefix, *harmonic_bond.params.shape)
        else:
            assert batch_du_dp is None
        if compute_du_dl:
            assert batch_du_dl.shape == (*shape_prefix,)
        else:
            assert batch_du_dl is None
        if compute_u:
            assert batch_u.shape == (*shape_prefix,)
        else:
            assert batch_u is None
        if compute_du_dx:
            np.testing.assert_array_equal(batch_du_dx, ref_du_dx)
        if compute_du_dp:
            np.testing.assert_array_equal(batch_du_dp, ref_du_dp)
        if compute_du_dl:
            np.testing.assert_array_equal(batch_du_dl, ref_du_dl)
        if compute_u:
            np.testing.assert_array_equal(batch_u, ref_u)


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

    harmonic_bond_1 = generic.HarmonicBond(random_bond_idxs())
    harmonic_bond_2 = generic.HarmonicBond(random_bond_idxs())

    params_1 = np.random.uniform(0, 1, size=(num_bonds, 2))
    params_2 = np.random.uniform(0, 1, size=(num_bonds, 2))

    return harmonic_bond_1, harmonic_bond_2, params_1, params_2, coords


def test_summed_potential(harmonic_bond_test_system):
    """Assert FanoutSummedPotential consistent with SummedPotential on
    a harmonic bond instance"""

    harmonic_bond_1, harmonic_bond_2, params_1, params_2, coords = harmonic_bond_test_system

    potential = generic.SummedPotential([harmonic_bond_1, harmonic_bond_2], [params_1, params_2])

    box = 3.0 * np.eye(3)
    lamb = 0.1

    params = np.concatenate((params_1.reshape(-1), params_2.reshape(-1)))

    for rtol, precision in [(1e-6, np.float32), (1e-10, np.float64)]:
        GradientTest().compare_forces_gpu_vs_reference(coords, params, box, [lamb], potential, rtol, precision)


def test_fanout_summed_potential_consistency(harmonic_bond_test_system):
    """Assert FanoutSummedPotential consistent with SummedPotential on
    a harmonic bond instance"""

    harmonic_bond_1, harmonic_bond_2, params, _, coords = harmonic_bond_test_system

    summed_potential = SummedPotential(
        [harmonic_bond_1.to_gpu(), harmonic_bond_2.to_gpu()],
        [params, params],
    )

    fanout_summed_potential = FanoutSummedPotential([harmonic_bond_1.to_gpu(), harmonic_bond_2.to_gpu()])

    box = 3.0 * np.eye(3)
    lamb = 0.1

    du_dx_ref, du_dps_ref, du_dl_ref, u_ref = summed_potential.unbound_impl(np.float32).execute(
        coords, [params, params], box, lamb
    )

    du_dx_test, du_dp_test, du_dl_test, u_test = fanout_summed_potential.unbound_impl(np.float32).execute(
        coords, params, box, lamb
    )

    np.testing.assert_array_equal(du_dx_ref, du_dx_test)
    np.testing.assert_allclose(np.sum(du_dps_ref, axis=0), du_dp_test, rtol=1e-8, atol=1e-8)
    assert du_dl_ref == du_dl_test
    assert u_ref == u_test
