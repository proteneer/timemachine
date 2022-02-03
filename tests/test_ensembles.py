from timemachine.md.ensembles import NVTEnsemble, NPTEnsemble
from simtk.unit import kelvin, atmosphere, kilojoule_per_mole, nanometer

from simtk import unit
import numpy as np

from timemachine.constants import ENERGY_UNIT, DISTANCE_UNIT


def _compute_reduced_potential(potential_energy, temperature, volume, pressure):
    """Convert potential energy into reduced potential.
    Copied from https://github.com/choderalab/openmmtools/blob/321b998fc5977a1f8893e4ad5700b1b3aef6101c/openmmtools/states.py#L1904-L1912
    """

    beta = 1.0 / (unit.BOLTZMANN_CONSTANT_kB * temperature)
    reduced_potential = potential_energy / unit.AVOGADRO_CONSTANT_NA
    if pressure is not None:
        reduced_potential += pressure * volume
    return beta * reduced_potential


def test_nvt():
    npt = NVTEnsemble(potential_energy=None, temperature=300 * kelvin)

    U = (-100 * kilojoule_per_mole).value_in_unit(ENERGY_UNIT)
    u_0 = npt.reduce(U)

    # check that reduced potential increases with increasing U
    assert npt.reduce(U + 10) > u_0

    # check that we agree with reference computation on some specific instances...
    n_trials = 100

    # random positive or negative
    potential_energies = np.random.randn(n_trials) * 100 * unit.kilojoule_per_mole

    # uniform between 100 and 400 Kelvin
    temperatures = (np.random.rand(n_trials) * 300 + 100) * unit.kelvin

    for (U, T) in zip(potential_energies, temperatures):
        ref = _compute_reduced_potential(U, T, None, None)
        nvt = NVTEnsemble(potential_energy=None, temperature=T)
        actual = nvt.reduce(U.value_in_unit(ENERGY_UNIT))
        np.testing.assert_almost_equal(actual, ref)


def test_npt():
    npt = NPTEnsemble(potential_energy=None, temperature=300 * kelvin, pressure=1 * atmosphere)

    U = (-100 * kilojoule_per_mole).value_in_unit(ENERGY_UNIT)
    volume = (4 * nanometer ** 3).value_in_unit(DISTANCE_UNIT ** 3)
    u_0 = npt.reduce(U, volume)

    # check that reduced potential increases with increasing U or volume
    assert npt.reduce(U + 10, volume) > u_0
    assert npt.reduce(U, volume + 1) > u_0

    # check that we agree with reference computation on some specific instances...
    n_trials = 100

    # random positive or negative
    potential_energies = np.random.randn(n_trials) * 100 * unit.kilojoule_per_mole

    # uniform between 100 and 400 Kelvin
    temperatures = (np.random.rand(n_trials) * 300 + 100) * unit.kelvin

    # box lengths uniform between 1 and 4 nm
    volumes = (np.random.rand(n_trials) * 3 + 1.0) ** 3 * unit.nanometer ** 3

    # uniform between 0.5 and 1.5 bar
    pressures = (np.random.rand(n_trials) + 0.5) * unit.bar

    for (U, T, V, P) in zip(potential_energies, temperatures, volumes, pressures):
        ref = _compute_reduced_potential(U, T, V, P)
        npt = NPTEnsemble(potential_energy=None, temperature=T, pressure=P)
        actual = npt.reduce(U.value_in_unit(ENERGY_UNIT), V.value_in_unit(DISTANCE_UNIT ** 3))

        np.testing.assert_almost_equal(actual, ref)
