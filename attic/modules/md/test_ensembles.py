import numpy as np
import pytest

from timemachine.constants import BOLTZ, AVOGADRO
from timemachine.md.ensembles import NPTEnsemble, NVTEnsemble

pytestmark = [pytest.mark.nogpu]


def _compute_reduced_potential(potential_energy, temperature, volume, pressure):
    """Convert potential energy into reduced potential.
    Copied from https://github.com/choderalab/openmmtools/blob/321b998fc5977a1f8893e4ad5700b1b3aef6101c/openmmtools/states.py#L1904-L1912
    and differs due to beta being defined differently in openmm and timemachine, with timemachine uses a beta reduced by AVOGADRO
    """

    kBT = BOLTZ * temperature
    beta = 1.0 / kBT
    reduced_potential = potential_energy
    if pressure is not None:
        reduced_potential += pressure * volume * AVOGADRO
    return beta * reduced_potential


def test_nvt():
    npt = NVTEnsemble(potential_energy=None, temperature=300)

    U = -100
    u_0 = npt.reduce(U)

    # check that reduced potential increases with increasing U
    assert npt.reduce(U + 10) > u_0

    # check that we agree with reference computation on some specific instances...
    n_trials = 100

    # random positive or negative
    potential_energies = np.random.randn(n_trials) * 100

    # uniform between 100 and 400 Kelvin
    temperatures = np.random.rand(n_trials) * 300 + 100

    for (U, T) in zip(potential_energies, temperatures):
        ref = _compute_reduced_potential(U, T, None, None)
        nvt = NVTEnsemble(potential_energy=None, temperature=T)
        actual = nvt.reduce(U)
        np.testing.assert_almost_equal(actual, ref)


def test_npt():
    npt = NPTEnsemble(potential_energy=None, temperature=300, pressure=1)

    U = -100
    volume = 4
    u_0 = npt.reduce(U, volume)

    # check that reduced potential increases with increasing U or volume
    assert npt.reduce(U + 10, volume) > u_0
    assert npt.reduce(U, volume + 1) > u_0

    # check that we agree with reference computation on some specific instances...
    n_trials = 100

    # random positive or negative
    potential_energies = np.random.randn(n_trials) * 100

    # uniform between 100 and 400 Kelvin
    temperatures = np.random.rand(n_trials) * 300 + 100

    # box lengths uniform between 1 and 4 nm
    volumes = (np.random.rand(n_trials) * 3 + 1.0) ** 3

    # uniform between 0.5 and 1.5 bar
    pressures = np.random.rand(n_trials) + 0.5

    for (U, T, V, P) in zip(potential_energies, temperatures, volumes, pressures):
        ref = _compute_reduced_potential(U, T, V, P)
        npt = NPTEnsemble(potential_energy=None, temperature=T, pressure=P)
        actual = npt.reduce(U, V)

        np.testing.assert_almost_equal(actual, ref)
