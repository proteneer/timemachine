import numpy as np
import pytest

from timemachine.lib import custom_ops, potentials

pytestmark = [pytest.mark.memcheck]


@pytest.fixture
def harmonic_bond():
    bond_idxs = np.array([[0, 1], [0, 2]], dtype=np.int32)
    params = np.ones(shape=(2, 2), dtype=np.float32)
    return potentials.HarmonicBond(bond_idxs).bind(params)


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
        potentials.SummedPotential([harmonic_bond], [])

    assert str(excinfo.value) == "number of potentials != number of parameter arrays"


def test_summed_potential_keeps_referenced_potentials_alive(harmonic_bond):
    sp = potentials.SummedPotential([harmonic_bond], [harmonic_bond.params]).bind(harmonic_bond.params)

    # segfaults if referenced potentials have been deallocated prematurely
    execute_bound_impl(sp.bound_impl(np.float32))


def test_summed_potential_get_potentials(harmonic_bond):
    impls = [harmonic_bond.unbound_impl(np.float32) for _ in range(2)]
    params_sizes = [len(harmonic_bond.params) for _ in range(2)]
    summed_impl = custom_ops.SummedPotential(impls, params_sizes)
    assert set(id(p) for p in summed_impl.get_potentials()) == set(id(p) for p in impls)
