from dataclasses import replace
from importlib import resources
from typing import List

import numpy as np
import pytest
from common import assert_energy_arrays_match, convert_quaternion_for_scipy, prepare_single_topology_initial_state
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp

from timemachine.constants import DEFAULT_PRESSURE, DEFAULT_TEMP
from timemachine.fe.free_energy import HostConfig, InitialState, MDParams, sample
from timemachine.fe.model_utils import apply_hmr, image_frame
from timemachine.fe.single_topology import SingleTopology
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops
from timemachine.md import builders
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.exchange.exchange_mover import BDExchangeMove as RefBDExchangeMove
from timemachine.md.exchange.exchange_mover import get_water_idxs, translate_coordinates
from timemachine.md.minimizer import check_force_norm
from timemachine.potentials import HarmonicBond, Nonbonded, SummedPotential
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


def verify_bias_deletion_moves(
    mol_groups: List,
    bdem,
    ref_bdem: RefBDExchangeMove,
    conf: NDArray,
    box: NDArray,
    total_num_proposals: int,
    proposals_per_move: int,
    rtol: float,
    atol: float,
):
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
            # The molecules should all be imaged in the home box
            np.testing.assert_allclose(image_frame(mol_groups, x_move, x_box), x_move)
            # Verify that the probabilities and per mol energies agree when we do accept moves
            # can only be done when we only attempt a single move per step
            before_log_weights = ref_bdem.batch_log_weights(last_conf, box)
            if num_moved == 1:
                # If only a single mol moved we can use incremental
                after_log_weights, tested = ref_bdem.batch_log_weights_incremental(
                    last_conf, x_box, idx, new_pos, before_log_weights
                )
                np.testing.assert_array_equal(tested, x_move)
            else:
                after_log_weights = ref_bdem.batch_log_weights(x_move, box)
            ref_log_prob = np.minimum(logsumexp(before_log_weights) - logsumexp(after_log_weights), 0.0)
            ref_prob = np.exp(ref_log_prob)
            assert np.isfinite(ref_prob) and ref_prob > 0.0
            # Only when proposals_per_move == 1 can we use the last log probability
            if proposals_per_move == 1:
                np.testing.assert_allclose(
                    np.exp(bdem.last_log_probability()),
                    ref_prob,
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"Step {step} failed",
                )
        elif num_moved == 0:
            np.testing.assert_array_equal(last_conf, x_move)
        assert proposals_per_move != 1 or num_moved <= 1, "More than one mol moved, something is wrong"

        last_conf = x_move
    print(f"Accepted {bdem.n_accepted()} of {total_num_proposals} moves")
    assert accepted > 0, "No moves were made, nothing was tested"
    if proposals_per_move == 1:
        assert bdem.n_accepted() == accepted
    else:
        assert bdem.n_accepted() >= accepted
    np.testing.assert_allclose(bdem.acceptance_fraction(), bdem.n_accepted() / bdem.n_proposed())


@pytest.mark.memcheck
@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_bd_exchange_validation(precision):
    N = 10
    beta = 1.2
    cutoff = 1.2
    seed = 2023
    klass = custom_ops.BDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.BDExchangeMove_f64

    rng = np.random.default_rng(2023)
    proposals_per_move = 1
    params = rng.random(size=(N, 4))

    # Test group indices verification
    group_idxs = []
    with pytest.raises(RuntimeError, match="must provide at least one molecule"):
        klass(N, group_idxs, params, DEFAULT_TEMP, beta, cutoff, seed, proposals_per_move, 1)

    # Second molecule is not contiguous with first
    group_idxs = [[0, 1, 2], [4, 5]]
    with pytest.raises(RuntimeError, match="Molecules are not contiguous: mol 1"):
        klass(N, group_idxs, params, DEFAULT_TEMP, beta, cutoff, seed, proposals_per_move, 1)

    group_idxs = [[0, 1, 2], [3, 4]]
    with pytest.raises(RuntimeError, match="only support running with mols with constant size, got mixed sizes"):
        klass(N, group_idxs, params, DEFAULT_TEMP, beta, cutoff, seed, proposals_per_move, 1)

    group_idxs = [[]]
    with pytest.raises(RuntimeError, match="must provide non-empty molecule indices"):
        klass(N, group_idxs, params, DEFAULT_TEMP, beta, cutoff, seed, proposals_per_move, 1)

    # Proposals must be non-zero
    with pytest.raises(RuntimeError, match="proposals per move must be greater than 0"):
        klass(N, group_idxs, params, DEFAULT_TEMP, beta, cutoff, seed, 0, 1)

    group_idxs = [[0], [1]]
    # Interval must be greater than zero
    with pytest.raises(RuntimeError, match="must provide interval greater than 0"):
        klass(N, group_idxs, params, DEFAULT_TEMP, beta, cutoff, seed, proposals_per_move, 0)

    with pytest.raises(RuntimeError, match="must provide batch size greater than 0"):
        klass(N, group_idxs, params, DEFAULT_TEMP, beta, cutoff, seed, proposals_per_move, 1, batch_size=-1)

    with pytest.raises(RuntimeError, match="number of proposals per move must be greater than batch size"):
        klass(
            N,
            group_idxs,
            params,
            DEFAULT_TEMP,
            beta,
            cutoff,
            seed,
            proposals_per_move,
            1,
            batch_size=proposals_per_move + 1,
        )

    klass(N, group_idxs, params, DEFAULT_TEMP, beta, cutoff, seed, proposals_per_move, 1)


@pytest.mark.memcheck
@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_bd_exchange_get_set_params(precision):
    N = 10
    beta = 1.2
    cutoff = 1.2
    seed = 2023
    klass = custom_ops.BDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.BDExchangeMove_f64

    rng = np.random.default_rng(2023)
    proposals_per_move = 1
    params = rng.random(size=(N, 4))

    group_idxs = [[0], [1]]

    exchange_move = klass(N, group_idxs, params, DEFAULT_TEMP, beta, cutoff, seed, proposals_per_move, 1)

    np.testing.assert_array_equal(params, exchange_move.get_params())

    params = rng.random(size=(N, 4))
    exchange_move.set_params(params)
    np.testing.assert_array_equal(params, exchange_move.get_params())

    with pytest.raises(RuntimeError, match="number of params don't match"):
        exchange_move.set_params(rng.random(size=(N, 3)))


@pytest.mark.memcheck
@pytest.mark.parametrize(
    "proposals_per_move,total_num_proposals,batch_size",
    [
        (1, 1, 1),
        (1, 10, 1),
        (10, 10, 10),
        pytest.param(1000, 10000, 1000, marks=pytest.mark.nightly(reason="slow")),
        pytest.param(1000, 10000, 333, marks=pytest.mark.nightly(reason="slow")),
    ],
)
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 5e-7, 5e-7), (np.float32, 1e-6, 2e-6)])
@pytest.mark.parametrize("seed", [2023])
def test_pair_of_waters_in_box(proposals_per_move, total_num_proposals, batch_size, precision, rtol, atol, seed):
    """Given two waters in a large box most moves should be accepted. This is a useful test for verifying memory doesn't leak"""
    ff = Forcefield.load_default()
    system, host_conf, _, _ = builders.build_water_system(1.0, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), host_conf.shape[0])

    # Get first two mols
    group_idxs = all_group_idxs[:2]

    box = np.eye(3) * 100.0

    # Re-image coords so that everything is imaged to begin with
    host_conf = image_frame(all_group_idxs, host_conf, box)

    conf_idxs = np.array(group_idxs).reshape(-1)
    conf = host_conf[conf_idxs]
    N = conf.shape[0]
    params = nb.params[conf_idxs]

    cutoff = nb.potential.cutoff
    klass = custom_ops.BDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.BDExchangeMove_f64

    bdem = klass(
        N,
        group_idxs,
        params,
        DEFAULT_TEMP,
        nb.potential.beta,
        cutoff,
        seed,
        proposals_per_move,
        1,
        batch_size=batch_size,
    )

    ref_bdem = RefBDExchangeMove(nb.potential.beta, cutoff, params, group_idxs, DEFAULT_TEMP)

    assert bdem.last_log_probability() == 0.0, "First log probability expected to be zero"
    verify_bias_deletion_moves(
        group_idxs, bdem, ref_bdem, conf, box, total_num_proposals, proposals_per_move, rtol, atol
    )
    assert bdem.n_proposed() == total_num_proposals


@pytest.mark.memcheck
@pytest.mark.parametrize(
    "proposals_per_move,total_num_proposals,batch_size",
    [(10000, 100000, 333)],
)
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 5e-7, 5e-7), (np.float32, 1e-6, 2e-6)])
@pytest.mark.parametrize("seed", [2023])
def test_sampling_single_water_in_bulk(
    proposals_per_move, total_num_proposals, batch_size, precision, rtol, atol, seed
):
    """Sample a single water in a box of water. Useful to verify that we are hitting the tail end of buffers"""
    ff = Forcefield.load_default()
    system, conf, box, _ = builders.build_water_system(2.5, ff.water_ff)
    box += np.diag([0.1, 0.1, 0.1])
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), conf.shape[0])

    # Randomly select a water to sample
    rng = np.random.default_rng(seed)
    water_idx = rng.integers(len(all_group_idxs))
    water_idxs = [all_group_idxs[water_idx]]

    # Re-image coords so that everything is imaged to begin with
    conf = image_frame(all_group_idxs, conf, box)

    N = conf.shape[0]
    params = nb.params

    cutoff = nb.potential.cutoff
    klass = custom_ops.BDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.BDExchangeMove_f64

    bdem = klass(
        N,
        water_idxs,
        params,
        DEFAULT_TEMP,
        nb.potential.beta,
        cutoff,
        seed,
        proposals_per_move,
        1,
        batch_size=batch_size,
    )

    ref_bdem = RefBDExchangeMove(nb.potential.beta, cutoff, params, water_idxs, DEFAULT_TEMP)

    assert bdem.last_log_probability() == 0.0, "First log probability expected to be zero"
    verify_bias_deletion_moves(
        all_group_idxs, bdem, ref_bdem, conf, box, total_num_proposals, proposals_per_move, rtol, atol
    )
    assert bdem.n_proposed() == total_num_proposals


@pytest.mark.parametrize("batch_size", [1, 200])
@pytest.mark.parametrize("precision", [np.float32])
@pytest.mark.parametrize("seed", [2023])
def test_bias_deletion_bulk_water_with_context(precision, seed, batch_size):
    ff = Forcefield.load_default()
    system, conf, box, _ = builders.build_water_system(4.0, ff.water_ff)
    box += np.diag([0.1, 0.1, 0.1])
    bps, masses = openmm_deserializer.deserialize_system(system, cutoff=1.2)
    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    bond_list = get_bond_list(bond_pot)
    all_group_idxs = get_group_indices(bond_list, conf.shape[0])

    # only act on waters
    water_idxs = get_water_idxs(all_group_idxs)

    dt = 2.5e-3

    masses = apply_hmr(masses, bond_list)

    bound_impls = []

    for potential in bps:
        bound_impls.append(potential.to_gpu(precision=np.float32).bound_impl)

    klass = custom_ops.BDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.BDExchangeMove_f64

    proposals_per_move = 2000
    interval = 100
    steps = interval * 40
    bdem = klass(
        conf.shape[0],
        water_idxs,
        nb.params,
        DEFAULT_TEMP,
        nb.potential.beta,
        nb.potential.cutoff,
        seed,
        proposals_per_move,
        interval,
        batch_size=batch_size,
    )

    intg = LangevinIntegrator(DEFAULT_TEMP, dt, 1.0, np.array(masses), seed).impl()

    barostat_interval = 5
    baro = MonteCarloBarostat(
        conf.shape[0],
        DEFAULT_PRESSURE,
        DEFAULT_TEMP,
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
        movers=[bdem, baro_impl],
    )
    xs, boxes = ctxt.multiple_steps(steps)
    assert bdem.n_proposed() == (steps // interval) * proposals_per_move
    assert bdem.n_accepted() > 0

    # Verify that the system is still stable
    for bp in bound_impls:
        du_dx, _ = bp.execute(xs[-1], boxes[-1], True, False)
        check_force_norm(-du_dx)


@pytest.mark.memcheck
@pytest.mark.parametrize(
    "proposals_per_move, batch_size",
    [(1, 1), (10, 1), (2, 2), (100, 100), pytest.param(1000, 333, marks=pytest.mark.nightly(reason="slow"))],
)
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("seed", [2023])
def test_bd_exchange_deterministic_moves(proposals_per_move, batch_size, precision, seed):
    """The exchange mover should accept every move and the results should be deterministic given the same seed and number of proposals per move


    There are three forms of determinism we require:
    * Constructing an exchange move produces the same results every time
    * Calling an exchange move with one proposals per move or K proposals per move produce the same state.
      * It is difficult to test each move when there are K proposals per move so we need to know that it matches the single proposals per move case
    * When we attempt K proposals in a batch (each proposal is made up of K proposals) it produces the same as the serial version
    """
    ff = Forcefield.load_default()
    system, conf, _, _ = builders.build_water_system(1.0, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    group_idxs = get_group_indices(get_bond_list(bond_pot), conf.shape[0])

    box = np.eye(3) * 100.0

    N = conf.shape[0]

    params = nb.params

    cutoff = nb.potential.cutoff
    klass = custom_ops.BDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.BDExchangeMove_f64

    # Reference that makes a single proposal per move
    bdem_a = klass(N, group_idxs, params, DEFAULT_TEMP, nb.potential.beta, cutoff, seed, 1, 1)
    # Test version that makes all proposals in a single move
    bdem_b = klass(
        N,
        group_idxs,
        params,
        DEFAULT_TEMP,
        nb.potential.beta,
        cutoff,
        seed,
        proposals_per_move,
        1,
        batch_size=batch_size,
    )

    iterative_moved_coords = conf.copy()
    for _ in range(proposals_per_move):
        iterative_moved_coords, _ = bdem_a.move(iterative_moved_coords, box)
        assert not np.all(conf == iterative_moved_coords)
    batch_moved_coords, _ = bdem_b.move(conf, box)

    # Typically this should accept at least half the number of moves as proposals
    assert bdem_a.n_accepted() >= max(proposals_per_move // 2, 1)
    assert bdem_a.n_proposed() == proposals_per_move
    assert bdem_a.n_accepted() == bdem_b.n_accepted()
    assert bdem_a.n_proposed() == bdem_b.n_proposed()

    # Moves should be deterministic regardless the number of proposals per move
    np.testing.assert_array_equal(iterative_moved_coords, batch_moved_coords)


@pytest.mark.parametrize("proposals_per_move, batch_size", [(2, 2), (100, 100), (512, 512), (2000, 1000)])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("seed", [2024])
def test_bd_exchange_deterministic_batch_moves(proposals_per_move, batch_size, precision, seed):
    """Verify that if we run with the same batch size but either call `move()` repeatedly or just
    increase the number of proposals per move in the constructor that the results should be identical
    """
    ff = Forcefield.load_default()
    system, conf, _, _ = builders.build_water_system(1.0, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    group_idxs = get_group_indices(get_bond_list(bond_pot), conf.shape[0])

    rng = np.random.default_rng(seed)

    box = np.eye(3) * 100.0

    N = conf.shape[0]

    params = nb.params

    cutoff = nb.potential.cutoff
    klass = custom_ops.BDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.BDExchangeMove_f64

    iterations = rng.integers(2, 5)

    # Reference that makes proposals_per_move proposals per move() call
    bdem_a = klass(
        N,
        group_idxs,
        params,
        DEFAULT_TEMP,
        nb.potential.beta,
        cutoff,
        seed,
        proposals_per_move,
        1,
        batch_size=batch_size,
    )
    # Test version that makes all proposals in a single move() call
    bdem_b = klass(
        N,
        group_idxs,
        params,
        DEFAULT_TEMP,
        nb.potential.beta,
        cutoff,
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


@pytest.mark.parametrize(
    "num_proposals_per_move,total_num_proposals,batch_size,box_size",
    [
        pytest.param(1, 4000, 1, 4.0, marks=pytest.mark.nightly(reason="slow")),
        (4000, 4000, 100, 4.0),
        (5000, 25000, 300, 4.0),
        # The 6.0nm box triggers a failure that would occur with systems of certain sizes, may be flaky in identifying issues
        pytest.param(1, 3000, 1, 6.0, marks=pytest.mark.nightly(reason="slow")),
    ],
)
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 5e-6, 5e-6), (np.float32, 1e-4, 2e-3)])
@pytest.mark.parametrize("seed", [2023])
def test_moves_in_a_water_box(
    num_proposals_per_move, total_num_proposals, batch_size, box_size, precision, rtol, atol, seed
):
    """Verify that the log acceptance probability between the reference and cuda implementation agree"""
    ff = Forcefield.load_default()
    system, conf, box, _ = builders.build_water_system(box_size, ff.water_ff)
    box += np.diag([0.1, 0.1, 0.1])
    bps, masses = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    bond_list = get_bond_list(bond_pot)
    group_idxs = get_group_indices(bond_list, conf.shape[0])

    N = conf.shape[0]

    # Re-image coords so that everything is imaged to begin with

    dt = 2.5e-3

    masses = apply_hmr(masses, bond_list)
    bound_impls = []

    for potential in bps:
        bound_impls.append(potential.to_gpu(precision=np.float32).bound_impl)

    intg = LangevinIntegrator(DEFAULT_TEMP, dt, 1.0, np.array(masses), seed).impl()

    barostat_interval = 5
    baro = MonteCarloBarostat(
        conf.shape[0],
        DEFAULT_PRESSURE,
        DEFAULT_TEMP,
        group_idxs,
        barostat_interval,
        seed,
    )

    ctxt = custom_ops.Context(
        conf,
        np.zeros_like(conf),
        box,
        intg,
        bound_impls,
        movers=[baro.impl(bound_impls)],
    )
    xs, boxes = ctxt.multiple_steps(10_000)
    conf = xs[-1]
    box = boxes[-1]
    for bp in bound_impls:
        du_dx, _ = bp.execute(conf, box, True, False)
        check_force_norm(-du_dx)

    conf = image_frame(group_idxs, conf, box)
    params = nb.params

    cutoff = nb.potential.cutoff
    klass = custom_ops.BDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.BDExchangeMove_f64

    bdem = klass(
        N,
        group_idxs,
        params,
        DEFAULT_TEMP,
        nb.potential.beta,
        cutoff,
        seed,
        num_proposals_per_move,
        1,
        batch_size=batch_size,
    )
    assert bdem.last_log_probability() == 0.0, "First log probability expected to be zero"

    ref_bdem = RefBDExchangeMove(nb.potential.beta, cutoff, params, group_idxs, DEFAULT_TEMP)

    iterations = 10
    # Up to some number of iterations of MD/MC are allowed before considering the test a failure
    # since there is relatively little space in which to insert a water. Requires MD/MC to ensure the
    # test is not flaky.
    for i in range(iterations):
        try:
            verify_bias_deletion_moves(
                group_idxs, bdem, ref_bdem, conf, box, total_num_proposals, num_proposals_per_move, rtol, atol
            )
            assert bdem.n_proposed() == (i + 1) * total_num_proposals
            # If we verified the target moves, we can exit
            return
        except AssertionError as e:
            if "No moves were made, nothing was tested" in str(e):
                xs, boxes = ctxt.multiple_steps(1000)
                conf = xs[-1]
                box = boxes[-1]
                conf = image_frame(group_idxs, conf, box)
                print("Running MD")
                continue
            # If an unexpect error was raised, re-raise the error
            raise
    assert False, f"No moves were made after {iterations}"


@pytest.mark.parametrize("num_particles", [3])
@pytest.mark.parametrize("proposals_per_move", [100, 1000])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("seed", [2023, 2024, 2025])
def test_compute_incremental_log_weights_match_initial_log_weights_when_recomputed(
    num_particles, proposals_per_move, precision, seed
):
    """Verify that the result of computing the weights using `compute_initial_log_weights` and the incremental log weights generated
    during proposals are identical.
    """
    assert (
        proposals_per_move > 1
    ), "If proposals per move is 1 then this isn't meaningful since the weights won't be incremental"
    rng = np.random.default_rng(seed)
    cutoff = 1.2
    beta = 2.0

    box_size = 1.0
    box = np.eye(3) * box_size
    conf = rng.random((num_particles, 3)) * box_size

    params = rng.random((num_particles, 4))
    params[:, 3] = 0.0  # Put them in the same plane

    group_idxs = [[x] for x in range(num_particles)]

    N = conf.shape[0]

    klass = custom_ops.BDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.BDExchangeMove_f64

    # Test version that makes all proposals in a single move
    bdem = klass(
        N,
        group_idxs,
        params,
        DEFAULT_TEMP,
        beta,
        cutoff,
        seed,
        proposals_per_move,
        1,
    )

    updated_coords, _ = bdem.move(conf, box)
    assert not np.all(updated_coords == conf)
    assert bdem.n_accepted() >= 1
    assert bdem.n_proposed() == proposals_per_move

    before_log_weights = bdem.get_before_log_weights()
    ref_log_weights = bdem.compute_initial_log_weights(updated_coords, box)
    # The before weights of the mover should identically match the weights if recomputed from scratch
    diff_idxs = np.argwhere(np.array(before_log_weights) != np.array(ref_log_weights))
    np.testing.assert_array_equal(before_log_weights, ref_log_weights, err_msg=f"idxs {diff_idxs} don't match")


@pytest.mark.parametrize(
    "batch_size,samples,box_size",
    [
        (1, 1000, 3.0),
        (2, 1000, 3.0),
        (1000, 10, 3.0),
    ],
)
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-5, 1e-5), (np.float32, 8e-4, 2e-3)])
@pytest.mark.parametrize("seed", [2023])
def test_compute_incremental_log_weights(batch_size, samples, box_size, precision, rtol, atol, seed):
    """Verify that the incremental weights computed are valid for different collections of rotations/translations"""
    proposals_per_move = batch_size  # Number doesn't matter here, we aren't calling move
    ff = Forcefield.load_default()
    system, conf, box, _ = builders.build_water_system(box_size, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    group_idxs = get_group_indices(get_bond_list(bond_pot), conf.shape[0])

    N = conf.shape[0]

    # Re-image coords so that everything is imaged to begin with
    conf = image_frame(group_idxs, conf, box)

    params = nb.params

    cutoff = nb.potential.cutoff
    klass = custom_ops.BDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.BDExchangeMove_f64

    bdem = klass(
        N, group_idxs, params, DEFAULT_TEMP, nb.potential.beta, cutoff, seed, proposals_per_move, 1, batch_size
    )

    ref_bdem = RefBDExchangeMove(nb.potential.beta, cutoff, params, group_idxs, DEFAULT_TEMP)

    rng = np.random.default_rng(seed)

    identity_mol_idxs = np.arange(len(group_idxs), dtype=np.int32)
    # Compute the initial reference log weights
    before_log_weights = ref_bdem.batch_log_weights(conf, box)
    for _ in range(samples):
        # Randomly select waters for sampling, no biasing here, just testing the incremental weights
        selected_mols = rng.choice(identity_mol_idxs, size=batch_size)
        quaternions = rng.normal(loc=0.0, scale=1.0, size=(batch_size, 4))
        # Scale the translations
        translations = rng.uniform(0, 1, size=(batch_size, 3)) * np.diagonal(box)

        test_weight_batches = bdem.compute_incremental_log_weights(conf, box, selected_mols, quaternions, translations)
        assert len(test_weight_batches) == batch_size
        for test_weights, selected_mol, quat, translation in zip(
            test_weight_batches, selected_mols, quaternions, translations
        ):
            moved_conf = conf.copy()
            rotation = Rotation.from_quat(convert_quaternion_for_scipy(quat))
            mol_idxs = group_idxs[selected_mol]
            mol_conf = moved_conf[mol_idxs]
            rotated_mol = rotation.apply(mol_conf)
            updated_mol_conf = translate_coordinates(rotated_mol, translation)

            ref_final_weights, trial_conf = ref_bdem.batch_log_weights_incremental(
                conf, box, selected_mol, updated_mol_conf, before_log_weights
            )
            moved_conf[mol_idxs] = updated_mol_conf
            np.testing.assert_equal(trial_conf, moved_conf)
            # Janky re-use of assert_energy_arrays_match which is for energies, but functions for any fixed point
            # Slightly reduced threshold to deal with these being weights
            assert_energy_arrays_match(
                np.array(ref_final_weights), np.array(test_weights), atol=atol, rtol=rtol, threshold=5e6
            )


@pytest.fixture(scope="module")
def hif2a_complex():
    seed = 2023
    ff = Forcefield.load_default()
    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        complex_system, conf, box, _, _ = builders.build_protein_system(str(path_to_pdb), ff.protein_ff, ff.water_ff)
    box += np.diag([0.1, 0.1, 0.1])
    bps, masses = openmm_deserializer.deserialize_system(complex_system, cutoff=1.2)
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    bond_list = get_bond_list(bond_pot)
    all_group_idxs = get_group_indices(bond_list, conf.shape[0])

    # Equilibrate the system a bit before hand, which reduces clashes in the system which results greater differences
    # between the reference and test case.
    dt = 2.5e-3
    temperature = DEFAULT_TEMP
    pressure = DEFAULT_PRESSURE

    masses = apply_hmr(masses, bond_list)
    intg = LangevinIntegrator(temperature, dt, 1.0, np.array(masses), seed).impl()

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
    ctxt.multiple_steps(10000)
    conf = ctxt.get_x_t()
    box = ctxt.get_box()
    for bp in bound_impls:
        du_dx, _ = bp.execute(conf, box, True, False)
        check_force_norm(-du_dx)
    return complex_system, conf, box


@pytest.mark.parametrize(
    "num_proposals_per_move,total_num_proposals,batch_size",
    [
        (10000, 200000, 1000),
    ],
)
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 5e-6, 5e-6), (np.float32, 1e-4, 2e-3)])
@pytest.mark.parametrize("seed", [2023])
def test_moves_with_complex(
    hif2a_complex, num_proposals_per_move, total_num_proposals, batch_size, precision, rtol, atol, seed
):
    complex_system, conf, box = hif2a_complex
    bps, masses = openmm_deserializer.deserialize_system(complex_system, cutoff=1.2)
    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    bond_list = get_bond_list(bond_pot)
    all_group_idxs = get_group_indices(bond_list, conf.shape[0])

    # only act on waters
    water_idxs = get_water_idxs(all_group_idxs)

    # Re-image coords so that everything is imaged to begin with
    conf = image_frame(all_group_idxs, conf, box)

    N = conf.shape[0]

    params = nb.params

    cutoff = nb.potential.cutoff
    klass = custom_ops.BDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.BDExchangeMove_f64

    bdem = klass(
        N,
        water_idxs,
        params,
        DEFAULT_TEMP,
        nb.potential.beta,
        cutoff,
        seed,
        num_proposals_per_move,
        1,
        batch_size=batch_size,
    )

    ref_bdem = RefBDExchangeMove(nb.potential.beta, cutoff, params, water_idxs, DEFAULT_TEMP)

    assert bdem.last_log_probability() == 0.0, "First log probability expected to be zero"
    verify_bias_deletion_moves(
        all_group_idxs, bdem, ref_bdem, conf, box, total_num_proposals, num_proposals_per_move, rtol, atol
    )
    assert bdem.n_proposed() == total_num_proposals


@pytest.fixture(scope="module")
def hif2a_rbfe_state() -> InitialState:
    seed = 2023
    ff = Forcefield.load_default()
    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        complex_system, complex_conf, box, _, num_water_atoms = builders.build_protein_system(
            str(path_to_pdb), ff.protein_ff, ff.water_ff
        )
    box += np.diag([0.1, 0.1, 0.1])
    host_config = HostConfig(complex_system, complex_conf, box, num_water_atoms)
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    st = SingleTopology(mol_a, mol_b, core, ff)

    initial_state = prepare_single_topology_initial_state(st, host_config)

    traj = sample(
        initial_state, MDParams(n_frames=1, n_eq_steps=0, steps_per_frame=10_000, seed=seed), max_buffer_frames=1
    )

    assert len(traj.frames) == 1
    x = traj.frames[0]
    b = traj.boxes[0]

    for bp in initial_state.potentials:
        du_dx, _ = bp.to_gpu(np.float32).bound_impl.execute(x, b, True, False)
        check_force_norm(-du_dx)
    return replace(initial_state, v0=traj.final_velocities, x0=x, box0=b)


@pytest.mark.parametrize(
    "num_proposals_per_move, total_num_proposals, batch_size",
    [
        (2000, 20000, 200),
    ],
)
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 5e-6, 5e-6), (np.float32, 1e-4, 2e-3)])
@pytest.mark.parametrize("seed", [2023])
def test_bd_moves_with_complex_and_ligand(
    hif2a_rbfe_state, num_proposals_per_move, total_num_proposals, batch_size, precision, rtol, atol, seed
):
    """Verify that when the water atoms are between the protein and ligand that the reference and cuda exchange mover agree"""
    initial_state = hif2a_rbfe_state

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

    N = conf.shape[0]

    cutoff = nb.potential.cutoff
    klass = custom_ops.BDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.BDExchangeMove_f64

    bdem = klass(
        N,
        water_idxs,
        water_params,
        DEFAULT_TEMP,
        nb.potential.beta,
        cutoff,
        seed,
        num_proposals_per_move,
        1,
        batch_size=batch_size,
    )

    ref_bdem = RefBDExchangeMove(nb.potential.beta, cutoff, water_params, water_idxs, DEFAULT_TEMP)

    assert bdem.last_log_probability() == 0.0, "First log probability expected to be zero"

    md_params = MDParams(n_frames=1, n_eq_steps=0, steps_per_frame=1000, seed=seed)

    iterations = 10
    # Up to some number of iterations of MD/MC are allowed before considering the test a failure
    # since there is relatively little space in which to insert a water. Requires MD/MC to ensure the
    # test is not flaky.
    for i in range(iterations):
        try:
            verify_bias_deletion_moves(
                all_group_idxs, bdem, ref_bdem, conf, box, total_num_proposals, num_proposals_per_move, rtol, atol
            )
            assert bdem.n_proposed() == (i + 1) * total_num_proposals
            # If we verified the target moves, we can exit
            return
        except AssertionError as e:
            if "No moves were made, nothing was tested" in str(e):
                traj = sample(
                    initial_state,
                    md_params,
                    max_buffer_frames=1,
                )

                assert len(traj.frames) == 1
                conf = traj.frames[0]
                box = traj.boxes[0]
                conf = image_frame(all_group_idxs, conf, box)
                print("Running MD")
                continue
            # If an unexpect error was raised, re-raise the error
            raise
    assert False, f"No moves were made after {iterations}"
