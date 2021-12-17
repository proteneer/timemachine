import pytest
import numpy as np

from timemachine.lib import custom_ops, potentials


@pytest.fixture
def harmonic_bond():
    bond_idxs = np.array([[0, 1], [0, 2]], np.int32)
    params = np.array([[1, 0], [0, 1]], dtype=np.float32)
    return potentials.HarmonicBond(bond_idxs).bind(params)


@pytest.fixture
def harmonic_bond_bound_impl(harmonic_bond):
    return custom_ops.BoundPotential(harmonic_bond.unbound_impl(np.float32), harmonic_bond.params)


def test_summed_potential_raises_on_inconsistent_lengths(harmonic_bond):

    with pytest.raises(ValueError) as excinfo:
        potentials.SummedPotential([harmonic_bond], [])

    assert str(excinfo.value) == "number of potentials != number of parameter arrays"


def test_bound_potential_get_potential(harmonic_bond):
    unbound_impl = harmonic_bond.unbound_impl(np.float32)
    bound_impl = custom_ops.BoundPotential(unbound_impl, harmonic_bond.params)

    assert unbound_impl is bound_impl.get_potential()


def test_bound_potential_preserves_reference_to_underlying_potential(harmonic_bond_bound_impl):
    assert isinstance(harmonic_bond_bound_impl.get_potential(), custom_ops.Potential)
