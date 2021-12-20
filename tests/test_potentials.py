import pytest
import numpy as np

from timemachine.lib import custom_ops, potentials


@pytest.fixture
def harmonic_bond():
    bond_idxs = np.array([[0, 1], [0, 2]], dtype=np.int32)
    params = np.ones(shape=(2, 2), dtype=np.float32)
    return potentials.HarmonicBond(bond_idxs).bind(params)


def test_bound_potential_keeps_referenced_potential_alive(harmonic_bond):
    bp = custom_ops.BoundPotential(harmonic_bond.unbound_impl(np.float32), harmonic_bond.params)

    coords = np.zeros(shape=(3, 3), dtype=np.float32)
    box = np.diag(np.ones(3))
    lam = 1

    # segfaults if referenced potential has been deallocated prematurely
    bp.execute(coords, box, lam)


def test_bound_potential_get_potential(harmonic_bond):
    unbound_impl = harmonic_bond.unbound_impl(np.float32)
    bound_impl = custom_ops.BoundPotential(unbound_impl, harmonic_bond.params)
    assert unbound_impl is bound_impl.get_potential()


def test_summed_potential_raises_on_inconsistent_lengths(harmonic_bond):
    with pytest.raises(ValueError) as excinfo:
        potentials.SummedPotential([harmonic_bond], [])

    assert str(excinfo.value) == "number of potentials != number of parameter arrays"


def test_summed_potential_keeps_referenced_potentials_alive(harmonic_bond):
    sp = potentials.SummedPotential([harmonic_bond], [harmonic_bond.params]).bind(harmonic_bond.params)

    coords = np.zeros(shape=(3, 3), dtype=np.float32)
    box = np.diag(np.ones(3))
    lam = 1

    # segfaults if referenced potentials have been deallocated prematurely
    sp.bound_impl(np.float32).execute(coords, box, lam)


def test_summed_potential_get_potentials(harmonic_bond):
    impls = [harmonic_bond.unbound_impl(np.float32) for _ in range(2)]
    params_sizes = [len(harmonic_bond.params) for _ in range(2)]
    summed_impl = custom_ops.SummedPotential(impls, params_sizes)
    assert set(id(p) for p in summed_impl.get_potentials()) == set(id(p) for p in impls)
