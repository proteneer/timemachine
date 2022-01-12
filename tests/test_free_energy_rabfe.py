import io
import numpy as np
from contextlib import redirect_stdout
from fe.free_energy_rabfe import (
    RABFEResult,
    setup_relative_restraints_by_distance,
    setup_relative_restraints_using_smarts,
    validate_lambda_schedule,
    interpolate_pre_optimized_protocol,
    construct_pre_optimized_absolute_lambda_schedule_solvent,
)
from fe.atom_mapping import CompareDistNonterminal
from fe.utils import get_romol_conf

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS

import pytest


def capture_print(closure):
    """capture anything printed when we call closure()"""

    f = io.StringIO()
    with redirect_stdout(f):
        closure()
    printed = f.getvalue()
    return printed


def test_rabfe_result_to_from_log():
    """assert equality after round-trip to/from preferred terminal log format"""

    result = RABFEResult("my mol", 1.0, 2.0, 3.0, 4.0)

    printed = capture_print(result.log)
    first_line = printed.splitlines()[0]

    reconstructed = RABFEResult.from_log(first_line)
    assert result == reconstructed


def test_setting_up_restraints_using_distance():
    seed = 814
    smi_a = "CCCONNN"
    smi_b = "CCCNNN"
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_a = Chem.AddHs(mol_a)

    mol_b = Chem.MolFromSmiles(smi_b)
    mol_b = Chem.AddHs(mol_b)
    for mol in [mol_a, mol_b]:
        AllChem.EmbedMolecule(mol, randomSeed=seed)

    mol_a_coords = get_romol_conf(mol_a)
    mol_b_coords = get_romol_conf(mol_b)

    core = setup_relative_restraints_by_distance(mol_a, mol_b)
    assert core.shape == (5, 2)

    # If we have a 0 cutoff, expect nothing to overlap
    core = setup_relative_restraints_by_distance(mol_a, mol_b, cutoff=0.0)
    assert core.shape == (0,)

    for cutoff in [0.08, 0.1, 0.2, 1.0]:
        core = setup_relative_restraints_by_distance(mol_a, mol_b, cutoff=cutoff)
        assert core.size > 0
        for a, b in core.tolist():
            assert np.linalg.norm(mol_a_coords[a] - mol_b_coords[b]) < cutoff

    # Adds seven hydrogen (terminal) atoms if allow terminal matches
    core = setup_relative_restraints_by_distance(mol_a, mol_b, terminal=True)
    assert core.shape == (12, 2)


def test_setting_up_restraints_using_smarts():
    seed = 814

    mcs_params = rdFMCS.MCSParameters()
    mcs_params.AtomTyper = CompareDistNonterminal()
    mcs_params.BondTyper = rdFMCS.BondCompare.CompareAny

    smi_a = "CCCONNN"
    smi_b = "CCCNNN"
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_a = Chem.AddHs(mol_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    mol_b = Chem.AddHs(mol_b)
    for mol in [mol_a, mol_b]:
        AllChem.EmbedMolecule(mol, randomSeed=seed)

    result = rdFMCS.FindMCS([mol_a, mol_b], mcs_params)

    core = setup_relative_restraints_using_smarts(mol_a, mol_b, result.smartsString)
    assert core.shape == (2, 2)


def test_validate_lambda_schedule():
    """check that assertions fail when they should"""
    for K in [50, 64]:
        good_lambda_schedule = np.linspace(0, 1, K)
        reversed_schedule = good_lambda_schedule[::-1]
        truncated_schedule = good_lambda_schedule[1:]

        validate_lambda_schedule(good_lambda_schedule, K)

        with pytest.raises(AssertionError):
            validate_lambda_schedule(reversed_schedule, K)

        with pytest.raises(AssertionError):
            validate_lambda_schedule(truncated_schedule, K - 1)

        with pytest.raises(AssertionError):
            validate_lambda_schedule(truncated_schedule, K)


def test_interpolate_pre_optimized_protocol():
    linear = np.linspace(0, 1, 50)
    nonlinear = np.linspace(0, 1, 64) ** 2

    for sched in [linear, nonlinear]:
        # recover ~exactly the initial schedule
        K = len(sched)
        sched_prime = interpolate_pre_optimized_protocol(sched, K)
        assert np.allclose(sched, sched_prime)

        # produce valid protocols when downsampling
        reduced = interpolate_pre_optimized_protocol(sched, K // 2)
        validate_lambda_schedule(reduced, K // 2)


def test_pre_optimized_solvent_decoupling_schedule():
    for K in [10, 50, 64, 128]:
        sched = construct_pre_optimized_absolute_lambda_schedule_solvent(K)
        validate_lambda_schedule(sched, K)
