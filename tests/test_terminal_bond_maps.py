"""
Assert invertibility for Interval-based bond length maps, and
assert accurate free energy estimates for hydrogen <-> halogen mutations using TerminalBondMaps
"""

import numpy as np
from jax import config, jit, vmap

config.update("jax_enable_x64", True)

from functools import partial

import pytest
from pymbar import bar, exp
from pymbar.mbar import MBAR
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine.constants import BOLTZ, DEFAULT_TEMP
from timemachine.fe.bar import DG_KEY
from timemachine.ff import Forcefield
from timemachine.maps.estimators import compute_mapped_reduced_work, compute_mapped_u_kn
from timemachine.maps.terminal_bonds import Interval, TerminalBondMap, TerminalMappableState, interval_map
from timemachine.md.enhanced import VacuumState, generate_ligand_samples

pytestmark = [pytest.mark.nogpu]


def test_invertibility_of_interval_maps():
    """Test that we can construct invertible maps between intervals on R^+"""

    # construct a bunch of random intervals
    np.random.seed(2022)
    states = [Interval(np.random.rand(), 1 + np.random.rand()) for _ in range(50)]

    # generate test points up to within eps of interval bounds,
    # (eps slightly > 0 to avoid spurious <=, >= assertion errors near bounds...)
    eps = 1e-8

    def construct_map(src: Interval, dst: Interval):
        return partial(interval_map, src_lb=src.lower, src_ub=src.upper, dst_lb=dst.lower, dst_ub=dst.upper)

    # for each pair of states, compute f, f_inv on a bunch of points, assert self-consistency
    for src in states:
        for dst in states:
            f = construct_map(src, dst)
            f_inv = construct_map(dst, src)

            # x in src
            xs = np.linspace(src.lower + eps, src.upper - eps, 1000)
            np.testing.assert_array_less(src.lower, xs, err_msg="x not in support of src!")
            np.testing.assert_array_less(xs, src.upper, err_msg="x not in support of src!")

            # y=f(x) in dst
            ys = vmap(f)(xs)
            assert ys.shape == xs.shape
            np.testing.assert_array_less(dst.lower, ys, err_msg="y not in support of dst!")
            np.testing.assert_array_less(ys, dst.upper, err_msg="y not in support of dst!")

            # x_=f_inv(f(x))
            xs_ = vmap(f_inv)(ys)
            np.testing.assert_array_less(src.lower, xs_, err_msg="f_inv(f(x)) not in support of src!")
            np.testing.assert_array_less(xs_, src.upper, err_msg="f_inv(f(x)) not in support of src!")
            np.testing.assert_allclose(xs_, xs, err_msg="f_inv(f(x)) != x!")


# utility functions for vacuum test system


def collect_samples(mol):
    ff = Forcefield.load_default()
    AllChem.EmbedMolecule(mol)
    samples = generate_ligand_samples(1000, mol, ff, DEFAULT_TEMP, 2022)[0][:, 0]
    return samples - samples[:, 0, np.newaxis]  # center first atom, for ease of visualization


def get_hb_params(mol):
    ff = Forcefield.load_default()
    params, bond_idxs = ff.hb_handle.parameterize(mol)
    return params, bond_idxs


def get_vacuum_u_fxn(mol, temperature=DEFAULT_TEMP):
    ff = Forcefield.load_default()
    U_fxn = jit(VacuumState(mol, ff).U_full)
    kBT = BOLTZ * temperature

    def u_fxn(xs):
        return vmap(U_fxn)(xs) / kBT

    return u_fxn


def test_on_methane():
    """Test TerminalBondMaps on methane <-> halogen-substituted methane.

    Note: The terminal bond map only adjusts bond lengths, but the system also contains changing angle terms,
    so the map does not reduce work standard deviation to 0.
    """
    np.random.seed(2022)

    parser_params = Chem.SmilesParserParams()
    parser_params.removeHs = False

    src_mol = Chem.MolFromSmiles("C([H])([H])([H])([H])", parser_params)
    dst_mol = Chem.MolFromSmiles("C(Br)(Cl)(F)(I)", parser_params)
    AllChem.EmbedMolecule(src_mol)
    AllChem.EmbedMolecule(dst_mol)

    # define src and dst states
    u_src = get_vacuum_u_fxn(src_mol)
    src_params, src_bond_idxs = get_hb_params(src_mol)
    src_state = TerminalMappableState.from_harmonic_bond_params(src_bond_idxs, src_params)

    u_dst = get_vacuum_u_fxn(dst_mol)
    dst_params, dst_bond_idxs = get_hb_params(dst_mol)
    dst_state = TerminalMappableState.from_harmonic_bond_params(dst_bond_idxs, dst_params)

    # collect samples in src state
    src_samples = collect_samples(src_mol)

    # raw works
    w_F = u_dst(src_samples) - u_src(src_samples)

    assert np.std(w_F) > 10.0

    # apply terminal bond map
    terminal_bond_map = TerminalBondMap.from_states(src_state, dst_state)
    mapped_w_F = compute_mapped_reduced_work(src_samples, u_src, u_dst, terminal_bond_map)

    assert np.std(mapped_w_F) < 1.0

    # also do reverse works
    dst_samples = collect_samples(dst_mol)
    inv_terminal_bond_map = TerminalBondMap.from_states(dst_state, src_state)
    mapped_w_R = compute_mapped_reduced_work(dst_samples, u_dst, u_src, inv_terminal_bond_map)

    assert np.std(mapped_w_R) < 1.0

    estimated_delta_f_forward = exp(mapped_w_F)[DG_KEY]

    estimated_delta_f_reverse = -exp(mapped_w_R)[DG_KEY]
    estimated_delta_f_bar = bar(mapped_w_F, mapped_w_R)[DG_KEY]

    # also plug into MBAR
    K = 2
    samples = [src_samples, dst_samples]
    N_k = [len(s) for s in samples]
    states = [src_state, dst_state]

    u_fxns = [u_src, u_dst]

    # confirm that TerminalBondMap.from_states(state, state) is identity
    src_identity_map = TerminalBondMap.from_states(src_state, src_state)
    _src_samples, _logdetjacs = src_identity_map(src_samples)
    np.testing.assert_allclose(_src_samples, src_samples)
    np.testing.assert_allclose(_logdetjacs, 0.0, atol=1e-12)

    # construct all pairs maps
    map_fxns = np.zeros((K, K), dtype=object)
    for i in range(K):
        for j in range(K):
            map_fxns[i, j] = TerminalBondMap.from_states(states[i], states[j])

    u_kn = compute_mapped_u_kn(samples, u_fxns, map_fxns)

    mbar = MBAR(u_kn, N_k)
    estimated_delta_f_mbar = mbar.f_k[1]

    estimates = np.array(
        [
            estimated_delta_f_forward,
            estimated_delta_f_reverse,
            estimated_delta_f_bar,
            estimated_delta_f_mbar,
        ]
    )
    # assert that all estimates are in agreement
    np.testing.assert_allclose(estimates[:3], estimates[-1], atol=1e-1)
