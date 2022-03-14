import numpy as np
import pytest

from timemachine.lib import custom_ops
from timemachine.lib.potentials import FanoutSummedPotential, HarmonicBond, SummedPotential


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


def test_fanout_summed_potential_consistency():
    """Assert FanoutSummedPotential consistent with SummedPotential on
    a harmonic bond instance"""

    np.random.seed(2022)

    num_atoms = 10
    num_bonds = 10
    box = 3.0 * np.eye(3)
    lamb = 0.1

    coords = np.random.uniform(0, 1, size=(num_atoms, 3)).astype(np.float32)

    def random_bond_idxs():
        return np.array(
            [np.random.choice(num_atoms, size=(2,), replace=False) for _ in range(num_bonds)], dtype=np.int32
        )

    harmonic_bond_1 = HarmonicBond(random_bond_idxs())
    harmonic_bond_2 = HarmonicBond(random_bond_idxs())

    params = np.random.uniform(0, 1, size=(num_bonds, 2))

    summed_potential = SummedPotential(
        [harmonic_bond_1, harmonic_bond_2],
        [params, params],
    )

    fanout_summed_potential = FanoutSummedPotential([harmonic_bond_1, harmonic_bond_2])

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
