import pytest
import numpy as np

from timemachine.lib import potentials


@pytest.fixture
def harmonic_bond():
    bond_idxs = np.array([[0, 1], [0, 2]], dtype=np.int32)
    params = np.ones(shape=(2, 2), dtype=np.float32)
    return potentials.HarmonicBond(bond_idxs).bind(params)


@pytest.fixture
def summed_potential(harmonic_bond):
    return potentials.SummedPotential([harmonic_bond], [harmonic_bond.params]).bind([harmonic_bond.params])


def test_summed_potential_raises_on_inconsistent_lengths(harmonic_bond):

    with pytest.raises(ValueError) as excinfo:
        potentials.SummedPotential([harmonic_bond], [])

    assert str(excinfo.value) == "number of potentials != number of parameter arrays"


def test_summed_potential_keeps_referenced_potentials_alive(summed_potential):
    coords = np.zeros(shape=(3, 3), dtype=np.float32)
    box = np.diag(np.ones(3))
    lam = 1

    # segfaults if Potentials referenced by summed_potential have been
    # deallocated prematurely
    summed_potential.bound_impl(np.float32).execute(coords, box, lam)
