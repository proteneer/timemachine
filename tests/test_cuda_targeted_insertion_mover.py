from dataclasses import replace
from importlib import resources
from typing import List

import numpy as np
import pytest
from common import prepare_single_topology_initial_state
from numpy.typing import NDArray
from scipy.special import logsumexp

from timemachine.constants import DEFAULT_ATOM_MAPPING_KWARGS, DEFAULT_PRESSURE, DEFAULT_TEMP
from timemachine.fe import atom_mapping
from timemachine.fe.free_energy import AbsoluteFreeEnergy, HostConfig, InitialState
from timemachine.fe.model_utils import apply_hmr, image_frame
from timemachine.fe.single_topology import SingleTopology
from timemachine.fe.topology import BaseTopology
from timemachine.fe.utils import read_sdf
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import MonteCarloBarostat, custom_ops
from timemachine.md import builders
from timemachine.md.barostat.utils import compute_box_volume, get_bond_list, get_group_indices
from timemachine.md.exchange.exchange_mover import TIBDExchangeMove as RefTIBDExchangeMove
from timemachine.md.exchange.exchange_mover import (
    compute_raw_ratio_given_weights,
    delta_r_np,
    get_water_groups,
    get_water_idxs,
)
from timemachine.md.minimizer import check_force_norm
from timemachine.potentials import HarmonicBond, Nonbonded, SummedPotential


def compute_ref_raw_log_prob(
    ref_exchange, water_idx, vi_mols, vj_mols, vol_i, vol_j, coords, box, sampled_mol_idx, new_coords
):
    """Modified from timemachine.md.exchange.exchange_mover.TIBDExchangeMove.swap_vi_into_vj
    to support not resampling"""
    coords = coords.copy()
    log_weights_before_full = ref_exchange.batch_log_weights(coords, box)
    log_weights_before = log_weights_before_full[vi_mols]

    # Verify that the probability of moving the specific water is relatively likely to move
    src_idx = np.argwhere(vi_mols == sampled_mol_idx).reshape(-1)
    log_probs_before = log_weights_before - logsumexp(log_weights_before)
    probs_before = np.exp(log_probs_before)
    median_probability = np.median(probs_before)
    reasonable_probability = min(median_probability, 0.01)
    assert (
        probs_before[src_idx] >= reasonable_probability
    ), f"Probability of moving water {src_idx} low {probs_before[src_idx]}, median is {median_probability}"

    vj_plus_one_idxs = np.concatenate([[water_idx], vj_mols])
    log_weights_after_full, trial_coords = ref_exchange.batch_log_weights_incremental(
        coords, box, water_idx, new_coords, log_weights_before_full
    )
    trial_coords = np.array(trial_coords)
    log_weights_after_full = np.array(log_weights_after_full)
    log_weights_after = log_weights_after_full[vj_plus_one_idxs]

    raw_log_p = compute_raw_ratio_given_weights(log_weights_before, log_weights_after, vi_mols, vj_mols, vol_i, vol_j)

    return trial_coords, raw_log_p


def verify_targeted_moves(
    mol_groups: List,
    bdem,
    ref_bdem: RefTIBDExchangeMove,
    conf: NDArray,
    box: NDArray,
    total_num_proposals: int,
    proposals_per_move: int,
    rtol: float,
    atol: float,
):
    assert bdem.last_log_probability() == 0.0, "First log probability expected to be zero"
    accepted = 0
    last_conf = conf
    for step in range(total_num_proposals // proposals_per_move):
        x_move, x_box = bdem.move(last_conf, box)
        # The box will never change
        np.testing.assert_array_equal(box, x_box)
        num_moved = 0
        new_pos = None
        idx = -1
        for i, mol_idxs in enumerate(ref_bdem.water_idxs_np):
            if not np.all(x_move[mol_idxs] == last_conf[mol_idxs]):
                num_moved += 1
                new_pos = x_move[mol_idxs]
                idx = i
        if num_moved > 0:
            print(f"Accepted {num_moved} moves on step {step}")
            accepted += 1
        if num_moved == 1:
            # The molecules should all be imaged in the home box
            np.testing.assert_allclose(image_frame(mol_groups, x_move, x_box), x_move)
            vol_inner = (4 / 3) * np.pi * ref_bdem.radius**3
            vol_outer = np.prod(np.diag(box)) - vol_inner

            center = np.mean(last_conf[ref_bdem.ligand_idxs], axis=0)

            inner_before, outer_before = get_water_groups(
                last_conf, box, center, ref_bdem.water_idxs_np, ref_bdem.radius
            )
            if idx in inner_before:
                vi_mols = inner_before
                vol_i = vol_inner
                vj_mols = outer_before
                vol_j = vol_outer
            else:
                vi_mols = outer_before
                vol_i = vol_outer
                vj_mols = inner_before
                vol_j = vol_inner
            tested, raw_ref_log_prob = compute_ref_raw_log_prob(
                ref_bdem, idx, vi_mols, vj_mols, vol_i, vol_j, last_conf, box, idx, new_pos
            )
            np.testing.assert_array_equal(tested, x_move)
            ref_prob = np.exp(raw_ref_log_prob)
            # positive inf is acceptable, negative inf is not
            assert not np.isnan(ref_prob) and ref_prob > 0.0
            # Verify that the probabilities and per mol energies agree when we do accept moves
            # can only be done when we only attempt a single move per step
            if proposals_per_move == 1:
                # Verify that the water moved from one region to another as expected
                inner_after, outer_after = get_water_groups(
                    x_move, box, center, ref_bdem.water_idxs_np, ref_bdem.radius
                )
                if idx in inner_before:
                    assert idx in outer_after
                else:
                    assert idx in inner_after
                raw_test_log_prob = bdem.last_raw_log_probability()
                # Verify that the raw, without min(x, 0.0), probabilities match coarsely
                np.testing.assert_allclose(
                    np.exp(raw_test_log_prob),
                    np.exp(raw_ref_log_prob),
                    rtol=1e-1,
                    atol=1e-3,
                    err_msg=f"Step {step} failed",
                )
                # Verify that the true log probabilities match with the specified rtol/atol
                np.testing.assert_allclose(
                    np.exp(min(raw_test_log_prob, 0.0)),
                    np.exp(min(raw_ref_log_prob, 0.0)),
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"Step {step} failed",
                )
        elif num_moved == 0:
            np.testing.assert_array_equal(last_conf, x_move)
        assert proposals_per_move != 1 or num_moved <= 1, "More than one mol moved, something is wrong"
        last_conf = x_move
    assert bdem.n_proposed() == total_num_proposals
    print(f"Accepted {accepted} of {total_num_proposals} moves")
    assert accepted > 0, "No moves were made, nothing was tested"
    if proposals_per_move == 1:
        np.testing.assert_allclose(bdem.acceptance_fraction(), accepted / total_num_proposals)
        assert bdem.n_accepted() == accepted
    else:
        assert bdem.n_accepted() >= accepted


@pytest.mark.memcheck
@pytest.mark.parametrize("seed", [2023, 2024])
@pytest.mark.parametrize("radius", [0.1, 0.5, 1.2, 2.0])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_inner_and_outer_water_groups(seed, radius, precision):
    rng = np.random.default_rng(seed)
    ff = Forcefield.load_default()
    system, coords, box, _ = builders.build_water_system(4.0, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), coords.shape[0])

    center_group_idx = rng.choice(np.arange(len(all_group_idxs)))

    center_group = all_group_idxs.pop(center_group_idx)

    group_idxs = np.delete(np.array(all_group_idxs).reshape(-1), center_group)
    group_idxs = group_idxs.reshape(len(all_group_idxs) - 1, 3)

    center = np.mean(coords[center_group], axis=0)

    ref_inner, ref_outer = get_water_groups(coords, box, center, group_idxs, radius)

    func = custom_ops.inner_and_outer_mols_f32
    if precision == np.float64:
        func = custom_ops.inner_and_outer_mols_f64

    inner_mol_idxs, outer_mol_idxs = func(center_group, coords, box, group_idxs, radius)
    inner_mol_idxs_second, outer_mol_idxs_second = func(center_group, coords, box, group_idxs, radius)
    np.testing.assert_array_equal(sorted(inner_mol_idxs), sorted(inner_mol_idxs_second))
    np.testing.assert_array_equal(sorted(outer_mol_idxs), sorted(outer_mol_idxs_second))

    assert len(inner_mol_idxs) + len(outer_mol_idxs) == len(group_idxs)
    np.testing.assert_equal(list(sorted(ref_inner)), list(sorted(inner_mol_idxs)))
    np.testing.assert_equal(list(sorted(ref_outer)), list(sorted(outer_mol_idxs)))


@pytest.mark.memcheck
@pytest.mark.parametrize("seed", [2023])
@pytest.mark.parametrize("n_translations", [1, 1000])
@pytest.mark.parametrize("radius", [1.0, 2.0])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_translations_inside_and_outside_sphere(seed, n_translations, radius, precision):
    rng = np.random.default_rng(seed)
    ff = Forcefield.load_default()
    system, coords, box, _ = builders.build_water_system(4.0, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), coords.shape[0])

    center_group_idx = rng.choice(np.arange(len(all_group_idxs)))

    center_group = all_group_idxs.pop(center_group_idx)

    center = np.mean(coords[center_group], axis=0)

    func = custom_ops.translations_inside_and_outside_sphere_host_f32
    if precision == np.float64:
        func = custom_ops.translations_inside_and_outside_sphere_host_f64

    translations_a = func(n_translations, box, center, radius, seed)
    translations_b = func(n_translations, box, center, radius, seed)
    assert translations_a.shape == (n_translations, 2, 3)
    # Bitwise deterministic with a provided seed
    np.testing.assert_array_equal(translations_a, translations_b)

    last_translation = None
    for i, translation in enumerate(translations_a):
        inner_translation, outer_translation = translation
        assert np.linalg.norm(delta_r_np(inner_translation, center, box)) < radius, str(i)
        assert np.linalg.norm(delta_r_np(outer_translation, center, box)) >= radius, str(i)
        if last_translation is not None:
            assert not np.all(last_translation == translation)
        last_translation = translation


@pytest.mark.memcheck
@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_tibd_exchange_validation(precision):
    N = 10
    beta = 1.2
    cutoff = 1.2
    seed = 2023
    klass = custom_ops.TIBDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.TIBDExchangeMove_f64

    rng = np.random.default_rng(2023)
    proposals_per_move = 1
    params = rng.random(size=(N, 4))

    ligand_idxs = np.array([0])
    radius = 1.0

    # Test group indices verification
    group_idxs = []
    with pytest.raises(RuntimeError, match="must provide at least one molecule"):
        klass(N, ligand_idxs, group_idxs, params, DEFAULT_TEMP, beta, cutoff, radius, seed, proposals_per_move, 1)

    # Second molecule is not contiguous with first
    group_idxs = [[0, 1, 2], [4, 5]]
    with pytest.raises(RuntimeError, match="Molecules are not contiguous: mol 1"):
        klass(N, ligand_idxs, group_idxs, params, DEFAULT_TEMP, beta, cutoff, radius, seed, proposals_per_move, 1)

    group_idxs = [[0, 1, 2], [3, 4]]
    with pytest.raises(RuntimeError, match="only support running with mols with constant size, got mixed sizes"):
        klass(N, ligand_idxs, group_idxs, params, DEFAULT_TEMP, beta, cutoff, radius, seed, proposals_per_move, 1)

    group_idxs = [[]]
    with pytest.raises(RuntimeError, match="must provide non-empty molecule indices"):
        klass(N, ligand_idxs, group_idxs, params, DEFAULT_TEMP, beta, cutoff, radius, seed, proposals_per_move, 1)

    # Proposals must be non-zero
    with pytest.raises(RuntimeError, match="proposals per move must be greater than 0"):
        klass(N, ligand_idxs, group_idxs, params, DEFAULT_TEMP, beta, cutoff, radius, seed, 0, 1)

    group_idxs = [[0]]
    # Radius must be non-zero
    with pytest.raises(RuntimeError, match="radius must be greater than 0.0"):
        klass(N, ligand_idxs, group_idxs, params, DEFAULT_TEMP, beta, cutoff, 0.0, seed, proposals_per_move, 1)

    # Interval must be greater than zero
    with pytest.raises(RuntimeError, match="must provide interval greater than 0"):
        klass(N, ligand_idxs, group_idxs, params, DEFAULT_TEMP, beta, cutoff, radius, seed, proposals_per_move, 0)

    with pytest.raises(RuntimeError, match="must provide batch size greater than 0"):
        klass(
            N,
            ligand_idxs,
            group_idxs,
            params,
            DEFAULT_TEMP,
            beta,
            cutoff,
            radius,
            seed,
            proposals_per_move,
            1,
            batch_size=0,
        )

    with pytest.raises(RuntimeError, match="number of proposals per move must be greater than batch size"):
        klass(
            N,
            ligand_idxs,
            group_idxs,
            params,
            DEFAULT_TEMP,
            beta,
            cutoff,
            radius,
            seed,
            proposals_per_move,
            1,
            batch_size=proposals_per_move + 1,
        )

    # Verify that if the box is too small this will trigger a failure
    # no such protection in the move_device call for performance reasons though an assert will be triggered
    box = np.eye(3) * 0.1
    radius = 1
    coords = rng.random((N, 3))
    mover = klass(N, ligand_idxs, group_idxs, params, DEFAULT_TEMP, beta, cutoff, radius, seed, proposals_per_move, 1)
    with pytest.raises(RuntimeError, match="volume of inner radius greater than box volume"):
        mover.move(coords, box)


@pytest.mark.memcheck
@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_tibd_exchange_get_set_params(precision):
    N = 10
    beta = 1.2
    cutoff = 1.2
    seed = 2023
    klass = custom_ops.TIBDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.TIBDExchangeMove_f64

    rng = np.random.default_rng(2023)
    proposals_per_move = 1
    params = rng.random(size=(N, 4))

    ligand_idxs = np.array([0])
    radius = 1.0
    group_idxs = [[0]]

    exchange_move = klass(
        N, ligand_idxs, group_idxs, params, DEFAULT_TEMP, beta, cutoff, radius, seed, proposals_per_move, 1
    )

    np.testing.assert_array_equal(params, exchange_move.get_params())

    params = rng.random(size=(N, 4))
    exchange_move.set_params(params)
    np.testing.assert_array_equal(params, exchange_move.get_params())

    with pytest.raises(RuntimeError, match="number of params don't match"):
        exchange_move.set_params(rng.random(size=(N, 3)))


@pytest.fixture(scope="module")
def brd4_rbfe_state() -> InitialState:
    seed = 2023
    # Use 0.5 lambda to ensure there is the most space in the binding pocket where both endstate's dummy atoms are
    # partially decoupled
    lamb = 0.5
    ff = Forcefield.load_default()
    # BRD4 is a known target that has waters in the binding site, use the structure with the water stripped from
    # the binding pocket
    with resources.path("timemachine.datasets.water_exchange", "brd4_no_water.pdb") as pdb_path:
        complex_system, complex_conf, box, _, num_water_atoms = builders.build_protein_system(
            str(pdb_path), ff.protein_ff, ff.water_ff
        )
    box += np.diag([0.1, 0.1, 0.1])
    with resources.path("timemachine.datasets.water_exchange", "brd4_pair.sdf") as ligand_path:
        mols = read_sdf(ligand_path)
    mol_a = mols[0]
    mol_b = mols[1]
    core = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)[0]
    host_config = HostConfig(complex_system, complex_conf, box, num_water_atoms)
    st = SingleTopology(mol_a, mol_b, core, ff)

    initial_state = prepare_single_topology_initial_state(st, host_config, lamb=lamb)
    conf = initial_state.x0

    bond_pot = next(bp for bp in initial_state.potentials if isinstance(bp.potential, HarmonicBond)).potential

    bond_list = get_bond_list(bond_pot)
    all_group_idxs = get_group_indices(bond_list, conf.shape[0])

    bps = initial_state.potentials
    masses = initial_state.integrator.masses

    # Equilibrate the system a bit before hand, which reduces clashes in the system which results greater differences
    # between the reference and test case.
    temperature = DEFAULT_TEMP
    pressure = DEFAULT_PRESSURE

    masses = apply_hmr(masses, bond_list)
    intg = initial_state.integrator.impl()

    bound_impls = []

    for potential in bps:
        bound_impls.append(potential.to_gpu(precision=np.float32).bound_impl)  # get the bound implementation

    barostat_interval = 5
    baro = MonteCarloBarostat(
        conf.shape[0],
        pressure,
        temperature,
        all_group_idxs,
        barostat_interval,
        seed,
    )
    baro_impl = baro.impl(bound_impls)

    ctxt = custom_ops.Context(
        conf,
        np.zeros_like(conf),
        box,
        intg,
        bound_impls,
        movers=[baro_impl],
    )
    xs, boxes = ctxt.multiple_steps(10_000)
    for bp in bound_impls:
        du_dx, _ = bp.execute(xs[-1], boxes[-1], True, False)
        check_force_norm(-du_dx)
    return replace(initial_state, v0=ctxt.get_v_t(), x0=xs[-1], box0=boxes[-1])


@pytest.mark.parametrize("radius", [0.4])
@pytest.mark.parametrize("moves", [10000])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 3e-5, 2e-5), (np.float32, 1e-4, 2e-3)])
@pytest.mark.parametrize("seed", [2024])
def test_targeted_insertion_buckyball_edge_cases(radius, moves, precision, rtol, atol, seed):
    """Test the edges cases of targeted insertion where the proposal probability isn't symmetric.

    Tests a single water and two waters being moved into an empty buckyball.
    """
    proposals_per_move = 1
    ff = Forcefield.load_precomputed_default()
    with resources.as_file(resources.files("timemachine.datasets.water_exchange")) as water_exchange:
        host_pdb = water_exchange / "bb_0_waters.pdb"
        mols = read_sdf(water_exchange / "bb_centered_espaloma.sdf")
        assert len(mols) == 1
        mol = mols[0]

    # Build the protein system using the solvent PDB for buckyball
    host_sys, host_conf, host_box, host_topology, num_water_atoms = builders.build_protein_system(
        str(host_pdb), ff.protein_ff, ff.water_ff
    )
    host_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    host_config = HostConfig(host_sys, host_conf, host_box, num_water_atoms)

    bt = BaseTopology(mol, ff)
    afe = AbsoluteFreeEnergy(mol, bt)
    # Fully embed the ligand
    potentials, params, combined_masses = afe.prepare_host_edge(ff.get_params(), host_config, 0.0)
    ligand_idxs = np.arange(num_water_atoms, num_water_atoms + mol.GetNumAtoms())

    conf = afe.prepare_combined_coords(host_config.conf)
    box = host_box

    bps = [pot.bind(p) for pot, p in zip(potentials, params)]
    summed_pot = next(bp.potential for bp in bps if isinstance(bp.potential, SummedPotential))
    nb = next(
        pot.bind(p) for pot, p in zip(summed_pot.potentials, summed_pot.params_init) if isinstance(pot, Nonbonded)
    )
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    bond_list = get_bond_list(bond_pot)
    all_group_idxs = get_group_indices(bond_list, conf.shape[0])

    # Select an arbitrary water, will be the only water considered for insertion/deletion
    water_idxs_single = [get_water_idxs(all_group_idxs)[-1]]
    water_idxs_double = get_water_idxs(all_group_idxs)[-2:]

    conf = image_frame(all_group_idxs, conf, box)

    N = conf.shape[0]

    params = nb.params

    cutoff = nb.potential.cutoff
    klass = custom_ops.TIBDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.TIBDExchangeMove_f64
    for water_idxs in [water_idxs_single, water_idxs_double]:
        bdem = klass(
            N,
            ligand_idxs,
            water_idxs,
            params,
            DEFAULT_TEMP,
            nb.potential.beta,
            cutoff,
            radius,
            seed,
            proposals_per_move,
            1,
        )

        ref_bdem = RefTIBDExchangeMove(nb.potential.beta, cutoff, params, water_idxs, DEFAULT_TEMP, ligand_idxs, radius)

        verify_targeted_moves(all_group_idxs, bdem, ref_bdem, conf, box, moves, proposals_per_move, rtol, atol)


@pytest.mark.parametrize("proposals_per_move, batch_size", [(20000, 1), (20000, 200)])
@pytest.mark.parametrize("radius", [1.3])
@pytest.mark.parametrize("precision", [np.float32])
@pytest.mark.parametrize("seed", [2023])
def test_targeted_insertion_brd4_rbfe_with_context(
    brd4_rbfe_state, proposals_per_move, batch_size, radius, precision, seed
):
    # Interval has to be large enough to resolve clashes in the MD steps
    interval = 800
    steps = interval * 10
    initial_state = brd4_rbfe_state

    conf = initial_state.x0
    box = initial_state.box0

    bps = initial_state.potentials
    summed_pot = next(bp for bp in initial_state.potentials if isinstance(bp.potential, SummedPotential))
    water_params = summed_pot.potential.params_init[0]
    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    bond_list = get_bond_list(bond_pot)
    all_group_idxs = get_group_indices(bond_list, conf.shape[0])

    # only act on waters
    water_idxs = get_water_idxs(all_group_idxs, ligand_idxs=initial_state.ligand_idxs)

    N = conf.shape[0]

    # Re-image coords so that everything is imaged to begin with
    conf = image_frame(all_group_idxs, conf, box)

    N = conf.shape[0]

    cutoff = nb.potential.cutoff
    klass = custom_ops.TIBDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.TIBDExchangeMove_f64

    bound_impls = []
    for potential in initial_state.potentials:
        bound_impls.append(potential.to_gpu(precision=np.float32).bound_impl)

    bdem = klass(
        N,
        initial_state.ligand_idxs,
        water_idxs,
        water_params,
        DEFAULT_TEMP,
        nb.potential.beta,
        cutoff,
        radius,
        seed,
        proposals_per_move,
        interval,
        batch_size=batch_size,
    )

    ctxt = custom_ops.Context(
        conf,
        np.zeros_like(conf),
        box,
        initial_state.integrator.impl(),
        bound_impls,
        movers=[bdem, initial_state.barostat.impl(bound_impls)],
    )
    xs, boxes = ctxt.multiple_steps(steps)
    assert bdem.n_proposed() == (steps // interval) * proposals_per_move
    assert bdem.n_accepted() > 0

    for bp in bound_impls:
        du_dx, _ = bp.execute(xs[-1], boxes[-1], True, False)
        check_force_norm(-du_dx)


@pytest.mark.parametrize("radius", [1.0])
@pytest.mark.parametrize("proposals_per_move, batch_size", [(2, 2), (100, 100), (512, 512), (2000, 1000)])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("seed", [2024])
def test_tibd_exchange_deterministic_batch_moves(radius, proposals_per_move, batch_size, precision, seed):
    """Verify that if we run with the same batch size but either call `move()` repeatedly or just
    increase the number of proposals per move in the constructor that the results should be identical
    """
    rng = np.random.default_rng(seed)
    ff = Forcefield.load_default()
    system, conf, _, _ = builders.build_water_system(1.0, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), conf.shape[0])

    group_idxs = all_group_idxs[1:]

    # Target the first water
    center_group = all_group_idxs[0]

    box = np.eye(3) * 100.0

    # Re-image coords so that everything is imaged to begin with
    conf = image_frame(all_group_idxs, conf, box)

    N = conf.shape[0]

    params = nb.params

    cutoff = nb.potential.cutoff

    klass = custom_ops.TIBDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.TIBDExchangeMove_f64

    iterations = rng.integers(2, 5)

    # Reference that makes proposals_per_move proposals per move() call
    bdem_a = klass(
        N,
        center_group,
        group_idxs,
        params,
        DEFAULT_TEMP,
        nb.potential.beta,
        cutoff,
        radius,
        seed,
        proposals_per_move,
        1,
    )

    bdem_b = klass(
        N,
        center_group,
        group_idxs,
        params,
        DEFAULT_TEMP,
        nb.potential.beta,
        cutoff,
        radius,
        seed,
        proposals_per_move * iterations,
        1,
        batch_size=batch_size,
    )

    iterative_moved_coords = conf.copy()
    for _ in range(iterations):
        iterative_moved_coords, _ = bdem_a.move(iterative_moved_coords, box)
        assert not np.all(conf == iterative_moved_coords)
    batch_moved_coords, _ = bdem_b.move(conf, box)

    assert bdem_a.n_accepted() > 0
    assert bdem_a.n_proposed() == proposals_per_move * iterations
    assert bdem_a.n_accepted() == bdem_b.n_accepted()
    assert bdem_a.n_proposed() == bdem_b.n_proposed()

    # Moves should be deterministic regardless the number of proposals per move
    np.testing.assert_array_equal(iterative_moved_coords, batch_moved_coords)


@pytest.mark.parametrize("radius", [0.4])
@pytest.mark.parametrize("proposals_per_move,batch_size", [(10000, 250)])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("seed", [2024])
def test_targeted_insertion_buckyball_determinism(radius, proposals_per_move, batch_size, precision, seed):
    """Test the edges cases of targeted insertion where the proposal probability isn't symmetric.

    Tests a single water and two waters being moved into an empty buckyball.
    """
    ff = Forcefield.load_precomputed_default()
    with resources.as_file(resources.files("timemachine.datasets.water_exchange")) as water_exchange:
        host_pdb = water_exchange / "bb_0_waters.pdb"
        mols = read_sdf(water_exchange / "bb_centered_espaloma.sdf")
        assert len(mols) == 1
        mol = mols[0]

    # Build the protein system using the solvent PDB for buckyball
    host_sys, host_conf, host_box, host_topology, num_water_atoms = builders.build_protein_system(
        str(host_pdb), ff.protein_ff, ff.water_ff
    )
    host_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    host_config = HostConfig(host_sys, host_conf, host_box, num_water_atoms)

    bt = BaseTopology(mol, ff)
    afe = AbsoluteFreeEnergy(mol, bt)
    # Fully embed the ligand
    potentials, params, combined_masses = afe.prepare_host_edge(ff.get_params(), host_config, 0.0)
    ligand_idxs = np.arange(num_water_atoms, num_water_atoms + mol.GetNumAtoms())

    conf = afe.prepare_combined_coords(host_config.conf)
    box = host_box

    bps = [pot.bind(p) for pot, p in zip(potentials, params)]
    summed_pot = next(bp.potential for bp in bps if isinstance(bp.potential, SummedPotential))
    nb = next(
        pot.bind(p) for pot, p in zip(summed_pot.potentials, summed_pot.params_init) if isinstance(pot, Nonbonded)
    )
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    bond_list = get_bond_list(bond_pot)
    all_group_idxs = get_group_indices(bond_list, conf.shape[0])
    water_idxs = get_water_idxs(all_group_idxs, ligand_idxs=ligand_idxs)

    conf = image_frame(all_group_idxs, conf, box)

    N = conf.shape[0]

    params = nb.params

    cutoff = nb.potential.cutoff
    klass = custom_ops.TIBDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.TIBDExchangeMove_f64

    # Reference that makes proposals_per_move proposals per move() call
    bdem_a = klass(
        N,
        ligand_idxs,
        water_idxs,
        params,
        DEFAULT_TEMP,
        nb.potential.beta,
        cutoff,
        radius,
        seed,
        proposals_per_move,
        1,
    )

    bdem_b = klass(
        N,
        ligand_idxs,
        water_idxs,
        params,
        DEFAULT_TEMP,
        nb.potential.beta,
        cutoff,
        radius,
        seed,
        proposals_per_move,
        1,
        batch_size=batch_size,
    )

    serially_moved_coords, _ = bdem_a.move(conf, box)
    assert not np.all(conf == serially_moved_coords)
    batch_moved_coords, _ = bdem_b.move(conf, box)

    assert bdem_a.n_accepted() > 0
    assert bdem_a.n_proposed() == proposals_per_move
    assert bdem_a.n_accepted() == bdem_b.n_accepted()
    assert bdem_a.n_proposed() == bdem_b.n_proposed()

    # Moves should be deterministic regardless the number of proposals per move
    np.testing.assert_array_equal(serially_moved_coords, batch_moved_coords)


@pytest.mark.parametrize("radius", [1.0])
@pytest.mark.parametrize("proposals_per_move, batch_size", [(5, 1), (100, 1), (12, 5), (120, 100), (1000, 1000)])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("seed", [2023])
def test_tibd_exchange_deterministic_moves(radius, proposals_per_move, batch_size, precision, seed):
    """Given a set of waters in a large box the exchange mover should nearly accept every move and the results should be deterministic
    if the seed and proposals per move are the same.


    There are three forms of determinism we require:
    * Constructing an exchange move produces the same results every time
    * Calling an exchange move with one proposals per move or K proposals per move produce the same state.
      * It is difficult to test each move when there are K proposals per move so we need to know that it matches the single proposals per move case
    * When batch size is greater than one (each batch is made up of K proposals) it produces the same result as the serial version (batch size == 1)
    """
    ff = Forcefield.load_default()
    system, conf, _, _ = builders.build_water_system(1.0, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), conf.shape[0])

    group_idxs = all_group_idxs[1:]

    # Target the first water
    center_group = all_group_idxs[0]

    box = np.eye(3) * 100.0

    # Re-image coords so that everything is imaged to begin with
    conf = image_frame(all_group_idxs, conf, box)

    N = conf.shape[0]

    params = nb.params

    cutoff = nb.potential.cutoff
    klass = custom_ops.TIBDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.TIBDExchangeMove_f64

    bdem_a = klass(
        N,
        center_group,
        group_idxs,
        params,
        DEFAULT_TEMP,
        nb.potential.beta,
        cutoff,
        radius,
        seed,
        1,
        1,
    )

    bdem_b = klass(
        N,
        center_group,
        group_idxs,
        params,
        DEFAULT_TEMP,
        nb.potential.beta,
        cutoff,
        radius,
        seed,
        proposals_per_move,
        1,
        batch_size=batch_size,
    )

    iterative_moved_coords = conf.copy()
    for _ in range(proposals_per_move):
        iterative_moved_coords, _ = bdem_a.move(iterative_moved_coords, box)
        assert not np.all(conf == iterative_moved_coords)  # We should move every time since its a single mol
    batch_moved_coords, _ = bdem_b.move(conf, box)
    assert bdem_a.n_accepted() > 0
    assert bdem_a.n_proposed() == proposals_per_move
    assert bdem_a.n_accepted() == bdem_b.n_accepted()
    assert bdem_a.n_proposed() == bdem_b.n_proposed()

    # Moves should be deterministic regardless the number of proposals per move
    np.testing.assert_array_equal(iterative_moved_coords, batch_moved_coords)
    if batch_size == 1:
        # Where the batch size is 1 the last log probabilities should match
        assert bdem_a.last_raw_log_probability() == bdem_b.last_raw_log_probability()


@pytest.mark.parametrize("radius", [1.2])
@pytest.mark.parametrize(
    "proposals_per_move,total_num_proposals,batch_size,,box_size",
    [
        (1, 500, 1, 4.0),
        (500, 10000, 250, 4.0),
        # The 5.7nm box triggers a failure that would occur with systems of certain sizes, may be flaky in identifying issues
        pytest.param(1, 5000, 1, 5.7, marks=pytest.mark.nightly(reason="slow")),
    ],
)
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 2e-5, 2e-5), (np.float32, 5e-3, 2e-3)])
@pytest.mark.parametrize("seed", [2023])
def test_targeted_moves_in_bulk_water(
    radius, proposals_per_move, total_num_proposals, batch_size, box_size, precision, rtol, atol, seed
):
    """Given bulk water molecules with one of them treated as the targeted region"""
    ff = Forcefield.load_default()
    system, conf, ref_box, topo = builders.build_water_system(box_size, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), conf.shape[0])

    center_group = all_group_idxs[-1]
    box = np.eye(3) * (radius * 2)
    # If box volume of system is larger than the box defined by radius, use that instead
    if compute_box_volume(ref_box) > compute_box_volume(box):
        box = ref_box

    # Re-image coords so that everything is imaged to begin with
    conf = image_frame(all_group_idxs, conf, box)

    group_idxs = all_group_idxs[:-1]

    N = conf.shape[0]

    params = nb.params

    cutoff = nb.potential.cutoff
    klass = custom_ops.TIBDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.TIBDExchangeMove_f64

    bdem = klass(
        N,
        center_group,
        group_idxs,
        params,
        DEFAULT_TEMP,
        nb.potential.beta,
        cutoff,
        radius,
        seed,
        proposals_per_move,
        1,
        batch_size=batch_size,
    )

    ref_bdem = RefTIBDExchangeMove(nb.potential.beta, cutoff, params, group_idxs, DEFAULT_TEMP, center_group, radius)
    verify_targeted_moves(
        all_group_idxs, bdem, ref_bdem, conf, box, total_num_proposals, proposals_per_move, rtol, atol
    )


@pytest.mark.parametrize("radius", [1.2])
@pytest.mark.parametrize(
    "proposals_per_move,total_num_proposals, batch_size",
    [
        pytest.param(1, 40000, 1, marks=pytest.mark.nightly(reason="slow")),
        (10000, 40000, 1),
        (10000, 40000, 250),
    ],
)
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 5e-6, 5e-6), (np.float32, 1e-4, 2e-3)])
@pytest.mark.parametrize("seed", [2023])
def test_moves_with_three_waters(
    radius, proposals_per_move, batch_size, total_num_proposals, precision, rtol, atol, seed
):
    """Given three water molecules with one of them treated as the targeted region."""
    ff = Forcefield.load_default()
    system, host_conf, _, _ = builders.build_water_system(1.0, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), host_conf.shape[0])

    # Get first two mols as the ones two move
    group_idxs = all_group_idxs[:2]

    # Third mol is the 'center' of the inner volume
    center_group = all_group_idxs[2]

    all_group_idxs = all_group_idxs[:3]

    box = np.eye(3) * 100.0

    conf_idxs = np.array(all_group_idxs).reshape(-1)
    conf = host_conf[conf_idxs]
    # Re-image coords so that everything is imaged to begin with
    conf = image_frame(all_group_idxs, conf, box)
    N = conf.shape[0]
    params = nb.params[conf_idxs]

    cutoff = nb.potential.cutoff
    klass = custom_ops.TIBDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.TIBDExchangeMove_f64

    bdem = klass(
        N,
        center_group,
        group_idxs,
        params,
        DEFAULT_TEMP,
        nb.potential.beta,
        cutoff,
        radius,
        seed,
        proposals_per_move,
        1,
        batch_size=batch_size,
    )

    ref_bdem = RefTIBDExchangeMove(nb.potential.beta, cutoff, params, group_idxs, DEFAULT_TEMP, center_group, radius)

    verify_targeted_moves(
        all_group_idxs, bdem, ref_bdem, conf, box, total_num_proposals, proposals_per_move, rtol, atol
    )


@pytest.mark.parametrize("radius", [0.95])
@pytest.mark.parametrize(
    "proposals_per_move,batch_size,total_num_proposals",
    [(10_000, 1, 200_000), (10_000, 200, 200_000)],
)
@pytest.mark.parametrize(
    "precision,rtol,atol",
    [pytest.param(np.float64, 2e-5, 2e-5, marks=pytest.mark.nightly(reason="slow")), (np.float32, 1e-4, 2e-3)],
)
@pytest.mark.parametrize("seed", [2023])
def test_targeted_moves_with_complex_and_ligand_in_brd4(
    brd4_rbfe_state, radius, proposals_per_move, batch_size, total_num_proposals, precision, rtol, atol, seed
):
    """Verify that when the water atoms are between the protein and ligand that the reference and cuda exchange mover agree.

    Uses BRD4 as the protein system as it has waters in the binding site
    """
    initial_state = brd4_rbfe_state

    conf = initial_state.x0
    box = initial_state.box0

    bps = initial_state.potentials

    summed_pot = next(bp for bp in bps if isinstance(bp.potential, SummedPotential))
    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    water_params = summed_pot.potential.params_init[0]

    bond_list = get_bond_list(bond_pot)
    all_group_idxs = get_group_indices(bond_list, conf.shape[0])

    # only act on waters
    water_idxs = get_water_idxs(all_group_idxs, ligand_idxs=initial_state.ligand_idxs)

    N = conf.shape[0]

    # Re-image coords so that everything is imaged to begin with
    conf = image_frame(all_group_idxs, conf, box)

    klass = custom_ops.TIBDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.TIBDExchangeMove_f64

    bdem = klass(
        N,
        initial_state.ligand_idxs,
        water_idxs,
        water_params,
        DEFAULT_TEMP,
        nb.potential.beta,
        nb.potential.cutoff,
        radius,
        seed,
        proposals_per_move,
        1,
        batch_size=batch_size,
    )

    ref_bdem = RefTIBDExchangeMove(
        nb.potential.beta,
        nb.potential.cutoff,
        water_params,
        water_idxs,
        DEFAULT_TEMP,
        initial_state.ligand_idxs,
        radius,
    )

    verify_targeted_moves(
        all_group_idxs, bdem, ref_bdem, conf, box, total_num_proposals, proposals_per_move, rtol, atol
    )
