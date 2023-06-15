import itertools

import jax
import numpy as np
import pytest

from timemachine.ff import Forcefield
from timemachine.md.enhanced import get_solvent_phase_system
from timemachine.potentials.executor import PotentialExecutor
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology

pytestmark = [pytest.mark.memcheck]


@pytest.fixture(scope="module")
def _hif2a_setup():
    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_default()
    # Don't minimize and have lamb = 1.0, as minimization may at some point rely on PotentialExecutor and want the
    # test to be free of using the PotentialExecutor
    unbound_potentials, sys_params, masses, coords, box = get_solvent_phase_system(
        mol_a, ff, 1.0, minimize_energy=False
    )
    return unbound_potentials, sys_params, masses, coords, box


@pytest.fixture()
def hif2a_system(_hif2a_setup):
    return _hif2a_setup


def execute_bound(bps, coords, box):
    du_dx_accum = np.zeros_like(coords)
    u_accum = 0.0
    for bp in bps:
        du_dx, u = bp.execute(coords, box)
        du_dx_accum += du_dx
        u_accum += u
    return du_dx_accum, u_accum


def execute_unbound(pots, params, coords, box):
    du_dx_accum = np.zeros_like(coords)
    du_dp_set = []
    u_accum = 0.0
    for pot, p in zip(pots, params):
        du_dx, du_dp, u = pot.execute(coords, p, box)
        du_dp_set.append(du_dp)
        du_dx_accum += du_dx
        u_accum += u
    return du_dx_accum, du_dp_set, u_accum


def test_potential_executor_bound_validation(hif2a_system):
    unbound_potentials, sys_params, masses, coords, box = hif2a_system

    bps = [pot.bind(p).to_gpu(np.float32).bound_impl for pot, p in zip(unbound_potentials, sys_params)]

    runner = PotentialExecutor(coords.shape[0])
    with pytest.raises(AssertionError, match="Must provide at least one potential"):
        runner.execute_bound([], coords, box)
    # The number of coords need to match original input
    with pytest.raises(AssertionError, match="Number of coordinates don't match"):
        runner.execute_bound(bps, coords[:3], box)


def test_potential_executor_unbound_validation(hif2a_system):
    unbound_potentials, sys_params, masses, coords, box = hif2a_system

    gpu_unbound = [pot.to_gpu(np.float32).unbound_impl for pot, p in zip(unbound_potentials, sys_params)]

    runner = PotentialExecutor(coords.shape[0])

    # Must pass at least one potential
    with pytest.raises(AssertionError, match="Must provide at least one potential"):
        runner.execute([], coords, sys_params, box)

    # The number of coords need to match original input
    with pytest.raises(AssertionError, match="Number of coordinates don't match"):
        runner.execute(gpu_unbound, coords[:3], sys_params, box)

    # The number of params don't match the number of potentials
    with pytest.raises(AssertionError, match="Number of potentials and params don't match"):
        runner.execute(gpu_unbound, coords, sys_params[:2], box)

    with pytest.raises(AssertionError, match="Number of potentials and params don't match"):
        runner.execute(gpu_unbound[:1], coords, sys_params, box)


def test_potential_executor_bound_selective(hif2a_system):
    unbound_potentials, sys_params, masses, coords, box = hif2a_system

    bps = [pot.bind(p).to_gpu(np.float32).bound_impl for pot, p in zip(unbound_potentials, sys_params)]

    runner = PotentialExecutor(coords.shape[0])
    for (compute_du_dx, compute_u) in itertools.product([False, True], repeat=2):
        if not compute_u and not compute_du_dx:
            with pytest.raises(RuntimeError):
                runner.execute_bound(bps, coords, box, compute_du_dx=compute_du_dx, compute_u=compute_u)
        else:
            test_du_dx, test_u = runner.execute_bound(
                bps, coords, box, compute_du_dx=compute_du_dx, compute_u=compute_u
            )
            if compute_du_dx:
                assert test_du_dx is not None
            else:
                assert test_du_dx is None
            if compute_u:
                assert test_u is not None
            else:
                assert test_u is None


def test_potential_executor_unbound_individual_gradients(hif2a_system):
    atol = 0.0
    rtol = 1e-8
    unbound_potentials, sys_params, masses, coords, box = hif2a_system

    gpu_unbound = [pot.to_gpu(np.float32).unbound_impl for pot, p in zip(unbound_potentials, sys_params)]

    ref_du_dx, ref_du_dps, ref_u = execute_unbound(gpu_unbound, sys_params, coords, box)

    runner = PotentialExecutor(coords.shape[0])

    grad_fn = jax.value_and_grad(runner.execute, argnums=(1,))

    test_u, (test_du_dx,) = grad_fn(gpu_unbound, coords, sys_params, box)
    np.testing.assert_allclose(ref_u, test_u, rtol=rtol, atol=atol)
    np.testing.assert_allclose(ref_du_dx, test_du_dx, rtol=rtol, atol=atol)

    grad_fn = jax.value_and_grad(runner.execute, argnums=(2,))

    test_u, (test_du_dps,) = grad_fn(gpu_unbound, coords, sys_params, box)
    np.testing.assert_allclose(ref_u, test_u, rtol=rtol, atol=atol)
    assert len(test_du_dps) == len(ref_du_dps)
    for ref, test in zip(ref_du_dps, test_du_dps):
        np.testing.assert_allclose(ref, test, rtol=rtol, atol=atol)

    # Should not be able to get the gradient with respect to the box
    grad_fn = jax.value_and_grad(runner.execute, argnums=(3,))

    with pytest.raises(RuntimeError, match="box derivatives not supported"):
        grad_fn(gpu_unbound, coords, sys_params, box)


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_potential_executor_bound(hif2a_system, precision):
    atol = 0.0
    rtol = 1e-8

    unbound_potentials, sys_params, masses, coords, box = hif2a_system

    bps = [pot.bind(p).to_gpu(precision).bound_impl for pot, p in zip(unbound_potentials, sys_params)]

    def verify_potential_list(bound_pots):

        ref_du_dx, ref_u = execute_bound(bound_pots, coords, box)

        runner = PotentialExecutor(coords.shape[0])
        test_du_dx, test_u = runner.execute_bound(bound_pots, coords, box)

        np.testing.assert_allclose(ref_du_dx, test_du_dx, rtol=rtol, atol=atol)
        np.testing.assert_allclose(ref_u, test_u, rtol=rtol, atol=atol)

    # Verify that passing lists of different potentials produce identical results
    verify_potential_list(bps)
    verify_potential_list(bps[:1])
    verify_potential_list(bps[2:])


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_potential_executor_unbound(hif2a_system, precision):
    atol = 0.0
    rtol = 1e-8

    unbound_potentials, sys_params, masses, coords, box = hif2a_system

    gpu_unbound = [pot.to_gpu(precision).unbound_impl for pot in unbound_potentials]

    def verify_potential_list(pots, params):

        ref_du_dx, ref_du_dps, ref_u = execute_unbound(pots, params, coords, box)

        runner = PotentialExecutor(coords.shape[0])
        test_u = runner.execute(pots, coords, params, box)

        np.testing.assert_allclose(ref_u, test_u, rtol=rtol, atol=atol)

        grad_fn = jax.value_and_grad(runner.execute, argnums=(1, 2))

        test_u_2, (test_du_dx, test_du_dps) = grad_fn(pots, coords, params, box)

        assert test_u == test_u_2

        np.testing.assert_allclose(ref_du_dx, test_du_dx, rtol=rtol, atol=atol)
        assert len(test_du_dps) == len(ref_du_dps)
        for ref, test in zip(ref_du_dps, test_du_dps):
            np.testing.assert_allclose(ref, test, rtol=rtol, atol=atol)

    verify_potential_list(gpu_unbound, sys_params)
    verify_potential_list(gpu_unbound[:1], sys_params[:1])
    verify_potential_list(gpu_unbound[2:], sys_params[2:])
