from dataclasses import replace
from importlib import resources

import numpy as np
import pytest
from common import prepare_single_topology_initial_state
from scipy.special import logsumexp

from timemachine.constants import DEFAULT_PRESSURE, DEFAULT_TEMP
from timemachine.fe.free_energy import HostConfig, InitialState
from timemachine.fe.model_utils import apply_hmr, image_frame
from timemachine.fe.single_topology import SingleTopology
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import MonteCarloBarostat, custom_ops
from timemachine.md import builders
from timemachine.md.barostat.utils import compute_box_volume, get_bond_list, get_group_indices
from timemachine.md.exchange.exchange_mover import TIBDExchangeMove as RefTIBDExchangeMove
from timemachine.md.exchange.exchange_mover import delta_r_np, get_water_groups
from timemachine.potentials import HarmonicBond, Nonbonded
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


def compute_ref_log_prob(ref_exchange, water_idx, vi_mols, vj_mols, vol_i, vol_j, coords, box, new_coords):
    """Modified from timemachine.md.exchange.exchange_mover.TIBDExchangeMove.swap_vi_into_vj
    to support not resampling"""
    coords = coords.copy()
    log_weights_before_full = ref_exchange.batch_log_weights(coords, box)
    log_weights_before = log_weights_before_full[vi_mols]

    vj_plus_one_idxs = np.concatenate([[water_idx], vj_mols])
    log_weights_after_full, trial_coords = ref_exchange.batch_log_weights_incremental(
        coords, box, water_idx, new_coords, log_weights_before_full
    )
    trial_coords = np.array(trial_coords)
    log_weights_after_full = np.array(log_weights_after_full)
    log_weights_after = log_weights_after_full[vj_plus_one_idxs]

    log_p_accept = min(0, logsumexp(log_weights_before) - logsumexp(log_weights_after) + np.log(vol_j) - np.log(vol_i))

    return trial_coords, log_p_accept


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
def test_translations_within_sphere(seed, n_translations, radius, precision):
    rng = np.random.default_rng(seed)
    ff = Forcefield.load_default()
    system, coords, box, _ = builders.build_water_system(4.0, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), coords.shape[0])

    center_group_idx = rng.choice(np.arange(len(all_group_idxs)))

    center_group = all_group_idxs.pop(center_group_idx)

    center = np.mean(coords[center_group], axis=0)

    func = custom_ops.translation_within_sphere_f32
    if precision == np.float64:
        func = custom_ops.translation_within_sphere_f64

    translations_a = func(n_translations, center, radius, seed)
    translations_b = func(n_translations, center, radius, seed)
    # Bitwise deterministic with a provided seed
    np.testing.assert_array_equal(translations_a, translations_b)

    last_translation = None
    for translation in translations_a:
        assert np.linalg.norm(delta_r_np(translation, center, box)) < radius
        if last_translation is not None:
            assert not np.all(last_translation == translation)


@pytest.mark.memcheck
@pytest.mark.parametrize("seed", [2023])
@pytest.mark.parametrize("n_translations", [1, 32])
@pytest.mark.parametrize("radius", [1.0, 2.0])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_translations_outside_sphere(seed, n_translations, radius, precision):
    rng = np.random.default_rng(seed)
    ff = Forcefield.load_default()
    system, coords, box, _ = builders.build_water_system(4.0, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), coords.shape[0])

    center_group_idx = rng.choice(np.arange(len(all_group_idxs)))

    center_group = all_group_idxs.pop(center_group_idx)

    center = np.mean(coords[center_group], axis=0)

    func = custom_ops.translation_outside_sphere_f32
    if precision == np.float64:
        func = custom_ops.translation_outside_sphere_f64

    translations_a = func(n_translations, center, box, radius, seed)
    translations_b = func(n_translations, center, box, radius, seed)
    # Bitwise deterministic with a provided seed
    np.testing.assert_array_equal(translations_a, translations_b)

    last_translation = None
    for i, translation in enumerate(translations_a):
        assert np.linalg.norm(delta_r_np(translation, center, box)) >= radius, str(i)
        if last_translation is not None:
            assert not np.all(last_translation == translation)


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
def hif2a_rbfe_state() -> InitialState:
    seed = 2023
    ff = Forcefield.load_default()
    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_ligand:
        complex_system, complex_conf, box, _, num_water_atoms = builders.build_protein_system(
            str(path_to_ligand), ff.protein_ff, ff.water_ff
        )

    host_config = HostConfig(complex_system, complex_conf, box, num_water_atoms)
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    st = SingleTopology(mol_a, mol_b, core, ff)

    initial_state = prepare_single_topology_initial_state(st, host_config)
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
    ctxt.multiple_steps(1000)
    conf = ctxt.get_x_t()
    box = ctxt.get_box()
    return replace(initial_state, v0=ctxt.get_v_t(), x0=conf, box0=box)


@pytest.mark.parametrize("radius", [2.0])
@pytest.mark.parametrize("precision", [np.float32])
@pytest.mark.parametrize("seed", [2023])
def test_targeted_insertion_hif2a_rbfe(hif2a_rbfe_state, radius, precision, seed):
    proposals_per_move = 10000
    interval = 100
    steps = 1000
    initial_state = hif2a_rbfe_state

    conf = initial_state.x0
    box = initial_state.box0

    bps = initial_state.potentials
    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    bond_list = get_bond_list(bond_pot)
    all_group_idxs = get_group_indices(bond_list, conf.shape[0])

    # only act on waters
    water_idxs = [group for group in all_group_idxs if len(group) == 3]

    N = conf.shape[0]

    # Re-image coords so that everything is imaged to begin with
    conf = image_frame(all_group_idxs, conf, box)

    N = conf.shape[0]

    params = nb.params

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
        params,
        DEFAULT_TEMP,
        nb.potential.beta,
        cutoff,
        radius,
        seed,
        proposals_per_move,
        interval,
    )

    ctxt = custom_ops.Context(
        conf,
        np.zeros_like(conf),
        box,
        initial_state.integrator.impl(),
        bound_impls,
        movers=[bdem, initial_state.barostat.impl(bound_impls)],
    )
    ctxt.multiple_steps(steps)
    assert bdem.n_proposed() == (steps // interval) * proposals_per_move
    assert bdem.n_accepted() > 0


@pytest.mark.parametrize("radius", [1.0])
@pytest.mark.parametrize("moves", [1, 100])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("seed", [2023])
def test_tibd_exchange_deterministic_moves(radius, moves, precision, seed):
    """Given one water the exchange mover should accept every move and the results should be deterministic"""
    ff = Forcefield.load_default()
    system, conf, _, _ = builders.build_water_system(1.0, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), conf.shape[0])

    # Get first two mols as the ones two move
    group_idxs = all_group_idxs[:2]

    center_group = all_group_idxs[2]

    all_group_idxs = all_group_idxs[:3]

    conf_idxs = np.array(all_group_idxs).reshape(-1)

    conf = conf[conf_idxs]

    # Set the two waters on top of each other
    conf[group_idxs[0], :] = conf[group_idxs[1], :]

    box = np.eye(3) * 100.0

    # Re-image coords so that everything is imaged to begin with
    conf = image_frame(all_group_idxs, conf, box)

    N = conf.shape[0]

    params = nb.params[conf_idxs]

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
        moves,
        1,
    )

    iterative_moved_coords = conf.copy()
    for _ in range(moves):
        iterative_moved_coords, _ = bdem_a.move(iterative_moved_coords, box)
    batch_moved_coords, _ = bdem_b.move(conf, box)
    # Moves should be deterministic regardless the number of steps taken per move
    np.testing.assert_array_equal(iterative_moved_coords, batch_moved_coords)
    assert bdem_a.n_accepted() > 0
    assert bdem_a.n_proposed() == moves
    assert bdem_a.n_accepted() == bdem_b.n_accepted()
    assert bdem_a.n_proposed() == bdem_b.n_proposed()


@pytest.mark.parametrize("radius", [1.2])
@pytest.mark.parametrize(
    "steps_per_move,moves",
    [(1, 500), (5000, 5000)],
)
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 5e-6, 5e-6), (np.float32, 1e-4, 2e-3)])
@pytest.mark.parametrize("seed", [2023])
def test_targeted_moves_in_bulk_water(radius, steps_per_move, moves, precision, rtol, atol, seed):
    """Given bulk water molecules with one of them treated as the targeted region"""
    ff = Forcefield.load_default()
    system, conf, ref_box, topo = builders.build_water_system(4.0, ff.water_ff)
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
        N, center_group, group_idxs, params, DEFAULT_TEMP, nb.potential.beta, cutoff, radius, seed, steps_per_move, 1
    )

    ref_bdem = RefTIBDExchangeMove(nb.potential.beta, cutoff, params, group_idxs, DEFAULT_TEMP, center_group, radius)

    assert bdem.last_log_probability() == 0.0, "First log probability expected to be zero"
    accepted = 0
    last_conf = conf
    for step in range(moves // steps_per_move):
        x_move, x_box = bdem.move(last_conf, box)
        # The box will never change
        np.testing.assert_array_equal(box, x_box)
        num_moved = 0
        new_pos = None
        idx = -1
        for i, mol_idxs in enumerate(group_idxs):
            if not np.all(x_move[mol_idxs] == last_conf[mol_idxs]):
                num_moved += 1
                new_pos = x_move[mol_idxs]
                idx = i
        if num_moved > 0:
            accepted += 1
            # The molecules should all be imaged in the home box
            np.testing.assert_allclose(image_frame(all_group_idxs, x_move, x_box), x_move)
            # Verify that the probabilities and per mol energies agree when we do accept moves
            # can only be done when we only attempt a single move per step
            if steps_per_move == 1:
                vol_inner = (4 / 3) * np.pi * radius**3
                vol_outer = np.prod(np.diag(box)) - vol_inner

                assert np.all(last_conf[center_group] == x_move[center_group])

                # Use the same inner/outer method that the Cuda version uses, can differ
                # slightly
                func = custom_ops.inner_and_outer_mols_f32
                if precision == np.float64:
                    func = custom_ops.inner_and_outer_mols_f64

                np.testing.assert_array_equal(last_conf[center_group], x_move[center_group])
                inner, outer = func(center_group, last_conf, box, group_idxs, radius)
                inner_after, outer_after = func(center_group, x_move, box, group_idxs, radius)
                if idx in inner:
                    assert idx in outer_after
                    vi_mols = inner
                    vol_i = vol_inner
                    vj_mols = outer
                    vol_j = vol_outer
                else:
                    assert idx in inner_after
                    vi_mols = outer
                    vol_i = vol_outer
                    vj_mols = inner
                    vol_j = vol_inner
                tested, ref_log_prob = compute_ref_log_prob(
                    ref_bdem, idx, vi_mols, vj_mols, vol_i, vol_j, last_conf, box, new_pos
                )
                ref_prob = np.exp(ref_log_prob)
                assert np.isfinite(ref_prob) and ref_prob > 0.0
                np.testing.assert_array_equal(tested, x_move)
                np.testing.assert_allclose(
                    np.exp(bdem.last_log_probability()),
                    np.exp(ref_log_prob),
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"Step {step} failed",
                )
        elif num_moved == 0:
            np.testing.assert_array_equal(last_conf, x_move)
        assert steps_per_move != 1 or num_moved <= 1, "More than one mol moved, something is wrong"

        last_conf = x_move
    assert bdem.n_proposed() == moves
    assert accepted > 0, "No moves were made, nothing was tested"
    if steps_per_move == 1:
        np.testing.assert_allclose(bdem.acceptance_fraction(), accepted / moves)
        assert bdem.n_accepted() == accepted
    else:
        assert bdem.n_accepted() >= accepted
        assert bdem.acceptance_fraction() >= accepted / moves


@pytest.mark.parametrize("radius", [1.2])
@pytest.mark.parametrize(
    "steps_per_move,moves",
    [(1, 500), (5000, 5000)],
)
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 5e-6, 5e-6), (np.float32, 1e-4, 2e-3)])
@pytest.mark.parametrize("seed", [2023])
def test_moves_with_three_waters(radius, steps_per_move, moves, precision, rtol, atol, seed):
    """Given three water molecules with one of them treated as the targeted region"""
    ff = Forcefield.load_default()
    system, conf, _, _ = builders.build_water_system(1.0, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), conf.shape[0])

    # Get first two mols as the ones two move
    group_idxs = all_group_idxs[:2]

    center_group = all_group_idxs[2]

    all_group_idxs = all_group_idxs[:3]

    conf_idxs = np.array(all_group_idxs).reshape(-1)

    conf = conf[conf_idxs]

    # Set the two waters on top of each other
    conf[group_idxs[0], :] = conf[group_idxs[1], :]

    box = np.eye(3) * 100.0

    # Re-image coords so that everything is imaged to begin with
    conf = image_frame(all_group_idxs, conf, box)

    N = conf.shape[0]

    params = nb.params[conf_idxs]

    cutoff = nb.potential.cutoff
    klass = custom_ops.TIBDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.TIBDExchangeMove_f64

    bdem = klass(
        N, center_group, group_idxs, params, DEFAULT_TEMP, nb.potential.beta, cutoff, radius, seed, steps_per_move, 1
    )

    ref_bdem = RefTIBDExchangeMove(nb.potential.beta, cutoff, params, group_idxs, DEFAULT_TEMP, center_group, radius)

    assert bdem.last_log_probability() == 0.0, "First log probability expected to be zero"
    accepted = 0
    last_conf = conf
    for step in range(moves // steps_per_move):
        x_move, x_box = bdem.move(last_conf, box)
        # The box will never change
        np.testing.assert_array_equal(box, x_box)
        num_moved = 0
        new_pos = None
        idx = -1
        for i, mol_idxs in enumerate(group_idxs):
            if not np.all(x_move[mol_idxs] == last_conf[mol_idxs]):
                num_moved += 1
                new_pos = x_move[mol_idxs]
                idx = i
        if num_moved > 0:
            accepted += 1
            # The molecules should all be imaged in the home box
            np.testing.assert_allclose(image_frame(all_group_idxs, x_move, x_box), x_move)
            # Verify that the probabilities and per mol energies agree when we do accept moves
            # can only be done when we only attempt a single move per step
            if steps_per_move == 1:
                vol_inner = (4 / 3) * np.pi * radius**3
                vol_outer = np.prod(np.diag(box)) - vol_inner

                center = np.mean(last_conf[center_group], axis=0)

                inner, outer = get_water_groups(last_conf, box, center, group_idxs, radius)
                if idx in inner:
                    vi_mols = inner
                    vol_i = vol_inner
                    vj_mols = outer
                    vol_j = vol_outer
                else:
                    vi_mols = outer
                    vol_i = vol_outer
                    vj_mols = inner
                    vol_j = vol_inner
                tested, ref_log_prob = compute_ref_log_prob(
                    ref_bdem, idx, vi_mols, vj_mols, vol_i, vol_j, last_conf, box, new_pos
                )
                ref_prob = np.exp(ref_log_prob)
                assert np.isfinite(ref_prob) and ref_prob > 0.0
                np.testing.assert_array_equal(tested, x_move)
                np.testing.assert_allclose(
                    np.exp(bdem.last_log_probability()),
                    np.exp(ref_log_prob),
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"Step {step} failed",
                )
        elif num_moved == 0:
            np.testing.assert_array_equal(last_conf, x_move)
        assert steps_per_move != 1 or num_moved <= 1, "More than one mol moved, something is wrong"

        last_conf = x_move
    assert bdem.n_proposed() == moves
    assert accepted > 0, "No moves were made, nothing was tested"
    if steps_per_move == 1:
        np.testing.assert_allclose(bdem.acceptance_fraction(), accepted / moves)
        assert bdem.n_accepted() == accepted
    else:
        assert bdem.n_accepted() >= accepted
        assert bdem.acceptance_fraction() >= accepted / moves


@pytest.mark.parametrize("radius", [2.0])
@pytest.mark.parametrize(
    "steps_per_move,moves",
    [(1, 12500), (12500, 12500)],
)
@pytest.mark.parametrize(
    "precision,rtol,atol",
    [pytest.param(np.float64, 5e-6, 5e-6, marks=pytest.mark.nightly(reason="slow")), (np.float32, 1e-4, 2e-3)],
)
@pytest.mark.parametrize("seed", [2023])
def test_moves_with_complex_and_ligand(hif2a_rbfe_state, radius, steps_per_move, moves, precision, rtol, atol, seed):
    """Verify that when the water atoms are between the protein and ligand that the reference and cuda exchange mover agree"""
    initial_state = hif2a_rbfe_state

    conf = initial_state.x0
    box = initial_state.box0

    bps = initial_state.potentials
    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    bond_list = get_bond_list(bond_pot)
    all_group_idxs = get_group_indices(bond_list, conf.shape[0])

    # only act on waters
    water_idxs = [group for group in all_group_idxs if len(group) == 3]

    N = conf.shape[0]

    # Re-image coords so that everything is imaged to begin with
    conf = image_frame(all_group_idxs, conf, box)

    N = conf.shape[0]

    params = nb.params

    cutoff = nb.potential.cutoff
    klass = custom_ops.TIBDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.TIBDExchangeMove_f64

    bdem = klass(
        N,
        initial_state.ligand_idxs,
        water_idxs,
        params,
        DEFAULT_TEMP,
        nb.potential.beta,
        cutoff,
        radius,
        seed,
        steps_per_move,
        1,
    )

    ref_bdem = RefTIBDExchangeMove(
        nb.potential.beta, cutoff, params, water_idxs, DEFAULT_TEMP, initial_state.ligand_idxs, radius
    )

    assert bdem.last_log_probability() == 0.0, "First log probability expected to be zero"
    accepted = 0
    last_conf = conf
    for step in range(moves // steps_per_move):
        x_move, x_box = bdem.move(last_conf, box)
        # The box will never change
        np.testing.assert_array_equal(box, x_box)
        num_moved = 0
        new_pos = None
        idx = -1
        for i, mol_idxs in enumerate(water_idxs):
            if not np.all(x_move[mol_idxs] == last_conf[mol_idxs]):
                num_moved += 1
                new_pos = x_move[mol_idxs]
                idx = i
        if num_moved > 0:
            accepted += 1
            # The molecules should all be imaged in the home box
            np.testing.assert_allclose(image_frame(all_group_idxs, x_move, x_box), x_move)
            # Verify that the probabilities and per mol energies agree when we do accept moves
            # can only be done when we only attempt a single move per step
            if steps_per_move == 1:
                vol_inner = (4 / 3) * np.pi * radius**3
                vol_outer = np.prod(np.diag(box)) - vol_inner

                center = np.mean(last_conf[initial_state.ligand_idxs], axis=0)
                inner, outer = get_water_groups(last_conf, box, center, water_idxs, radius)
                if idx in inner:
                    vi_mols = inner
                    vol_i = vol_inner
                    vj_mols = outer
                    vol_j = vol_outer
                else:
                    vi_mols = outer
                    vol_i = vol_outer
                    vj_mols = inner
                    vol_j = vol_inner
                tested, ref_log_prob = compute_ref_log_prob(
                    ref_bdem, idx, vi_mols, vj_mols, vol_i, vol_j, last_conf, box, new_pos
                )
                np.testing.assert_array_equal(tested, x_move)
                ref_prob = np.exp(ref_log_prob)
                assert np.isfinite(ref_prob) and ref_prob > 0.0
                np.testing.assert_allclose(
                    np.exp(bdem.last_log_probability()),
                    np.exp(ref_log_prob),
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"Step {step} failed",
                )
        elif num_moved == 0:
            np.testing.assert_array_equal(last_conf, x_move)
        assert steps_per_move != 1 or num_moved <= 1, "More than one mol moved, something is wrong"

        last_conf = x_move
    assert bdem.n_proposed() == moves
    assert accepted > 0, "No moves were made, nothing was tested"
    if steps_per_move == 1:
        np.testing.assert_allclose(bdem.acceptance_fraction(), accepted / moves)
        assert bdem.n_accepted() == accepted
    else:
        assert bdem.n_accepted() >= accepted
        assert bdem.acceptance_fraction() >= accepted / moves
