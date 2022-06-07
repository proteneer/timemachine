import functools

import numpy as np
import pytest

from timemachine.constants import DEFAULT_FF
from timemachine.datasets import fetch_freesolv
from timemachine.fe import utils
from timemachine.fe.functional import construct_differentiable_interface_fast
from timemachine.ff import Forcefield
from timemachine.ff.handlers import AM1CCCHandler
from timemachine.md import enhanced
from timemachine.md.enhanced import identify_rotatable_bonds


def test_identify_rotatable_bonds_runs_on_freesolv():
    """pass if no runtime errors are encountered"""
    mols = fetch_freesolv()

    for mol in mols:
        _ = identify_rotatable_bonds(mol)


@pytest.fixture
def ff_pair():
    ff0 = Forcefield.load_from_file(DEFAULT_FF)
    ff1 = Forcefield.load_from_file(DEFAULT_FF)

    # Modify the charge parameters for ff1
    for h, p in zip(ff1.get_ordered_handles(), ff1.get_ordered_params()):
        if isinstance(h, AM1CCCHandler):
            p += 1.0
    return ff0, ff1


def test_get_solvent_phase_system_parameter_changes(ff_pair):
    ff0, ff1 = ff_pair
    mol = fetch_freesolv()[0]

    ubps, params, m, coords, box = enhanced.get_solvent_phase_system_parameter_changes(
        mol, ff0=ff0, ff1=ff1, minimize_energy=False
    )
    U_ff = construct_differentiable_interface_fast(ubps, params)

    ubps0, params0, m0, coords0, box0 = enhanced.get_solvent_phase_system(mol, ff0, minimize_energy=False)
    U0 = construct_differentiable_interface_fast(ubps0, params0)

    ubps1, params1, m1, coords1, box1 = enhanced.get_solvent_phase_system(mol, ff1, minimize_energy=False)
    U1 = construct_differentiable_interface_fast(ubps1, params1)

    u_ff0 = U_ff(coords, params, box, lam=0)
    u_ff1 = U_ff(coords, params, box, lam=1)

    u0 = U0(coords0, params0, box0, lam=0)
    u1 = U1(coords1, params1, box1, lam=0)

    # Check that the endstate energies are consistent
    assert pytest.approx(u_ff0) == u0
    assert pytest.approx(u_ff1) == u1

    # Check that the masses are consistent
    assert pytest.approx(m) == m0
    assert pytest.approx(m) == m1

    # Check that the box is consistent
    assert pytest.approx(box) == box0
    assert pytest.approx(box) == box1

    # Check that the coords is consistent
    assert pytest.approx(coords) == coords0
    assert pytest.approx(coords) == coords1


def test_get_vacuum_phase_system_parameter_changes(ff_pair):
    ff0, ff1 = ff_pair
    get_system_fxn = functools.partial(enhanced.get_vacuum_phase_system_parameter_changes, ff0=ff0, ff1=ff1)

    mol = fetch_freesolv()[0]

    ubps, params, m, coords = get_system_fxn(mol)
    U_ff = construct_differentiable_interface_fast(ubps, params)

    U0 = enhanced.VacuumState(mol, ff0).U_full
    U1 = enhanced.VacuumState(mol, ff1).U_full

    box = np.eye(3) * 1000.0
    u_ff0 = U_ff(coords, params, box, lam=0)
    u_ff1 = U_ff(coords, params, box, lam=1)

    u0 = U0(coords)
    u1 = U1(coords)

    # Check that the endstate energies are consistent
    assert pytest.approx(u_ff0) == u0
    assert pytest.approx(u_ff1) == u1

    # Check that the masses are consistent
    assert pytest.approx(m) == utils.get_mol_masses(mol)

    # Check that the coords is consistent
    assert pytest.approx(coords) == utils.get_romol_conf(mol)
