import numpy as np
import pytest
from scipy.special import logsumexp

from timemachine.constants import DEFAULT_TEMP
from timemachine.fe.model_utils import image_frame
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import custom_ops
from timemachine.md import builders
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.exchange.exchange_mover import BDExchangeMove as RefBDExchangeMove
from timemachine.potentials import HarmonicBond, Nonbonded


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
    params = rng.random(size=(10, 4))

    # Test group indices verification
    group_idxs = []
    with pytest.raises(RuntimeError, match="must provide at least one molecule"):
        klass(N, group_idxs, params, DEFAULT_TEMP, beta, cutoff, seed, proposals_per_move)

    # Molecule that doesn't start from 0
    group_idxs = [[3, 4, 5]]
    with pytest.raises(RuntimeError, match="Molecules are not contiguous: mol 0"):
        klass(N, group_idxs, params, DEFAULT_TEMP, beta, cutoff, seed, proposals_per_move)

    # Second molecule is not contiguous with first
    group_idxs = [[0, 1, 2], [4, 5]]
    with pytest.raises(RuntimeError, match="Molecules are not contiguous: mol 1"):
        klass(N, group_idxs, params, DEFAULT_TEMP, beta, cutoff, seed, proposals_per_move)

    group_idxs = [[0, 1, 2], [3, 4]]
    with pytest.raises(RuntimeError, match="only support running with mols with constant size, got mixed sizes"):
        klass(N, group_idxs, params, DEFAULT_TEMP, beta, cutoff, seed, proposals_per_move)

    group_idxs = [[]]
    with pytest.raises(RuntimeError, match="must provide non-empty molecule indices"):
        klass(N, group_idxs, params, DEFAULT_TEMP, beta, cutoff, seed, proposals_per_move)

    # Proposals must be non-zero
    with pytest.raises(RuntimeError, match="proposals per move must be greater than 0"):
        klass(N, group_idxs, params, DEFAULT_TEMP, beta, cutoff, seed, 0)


@pytest.mark.memcheck
@pytest.mark.parametrize("moves", [1, 2, 10])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 5e-7, 5e-7), (np.float32, 1e-6, 2e-6)])
@pytest.mark.parametrize("seed", [2023])
def test_two_clashy_water_moves(moves, precision, rtol, atol, seed):
    """Given two waters directly on top of each other in a box, the exchange mover should accept almost any move"""
    ff = Forcefield.load_default()
    system, conf, _, _ = builders.build_water_system(1.0, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), conf.shape[0])

    # Get first two mols
    group_idxs = all_group_idxs[:2]

    conf_idxs = np.array(group_idxs).reshape(-1)

    conf = conf[conf_idxs]
    # Set the two waters ontop of each other
    conf[group_idxs[0], :] = conf[group_idxs[1], :]

    box = np.eye(3) * 100.0

    # Re-image coords so that everything is imaged to begin with
    conf = image_frame(group_idxs, conf, box)

    N = conf.shape[0]

    params = nb.params[conf_idxs]

    cutoff = nb.potential.cutoff
    klass = custom_ops.BDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.BDExchangeMove_f64

    proposals_per_move = 1
    bdem = klass(N, group_idxs, params, DEFAULT_TEMP, nb.potential.beta, cutoff, seed, proposals_per_move)

    ref_bdem = RefBDExchangeMove(nb.potential.beta, cutoff, params, group_idxs, DEFAULT_TEMP)

    assert bdem.last_log_probability() == 0.0, "First log probability expected to be zero"
    last_conf = conf
    accepted = 0
    for _ in range(moves):
        before_log_weights = ref_bdem.batch_log_weights(last_conf, box)
        x_move, x_box = bdem.move(last_conf, box)
        after_log_weights = ref_bdem.batch_log_weights(x_move, x_box)
        # The box will never change
        np.testing.assert_array_equal(box, x_box)
        num_moved = 0
        for mol_idxs in group_idxs:
            if not np.all(x_move[mol_idxs] == last_conf[mol_idxs]):
                num_moved += 1
        if num_moved > 0 and proposals_per_move == 1:
            accepted += 1
            # Verify that the probabilities agree when we do accept moves
            np.testing.assert_allclose(
                np.exp(bdem.last_log_probability()),
                np.exp(np.minimum(logsumexp(before_log_weights) - logsumexp(after_log_weights), 0.0)),
                rtol=rtol,
                atol=atol,
            )
            # The molecules should all be imaged in the home box
            np.testing.assert_allclose(image_frame(group_idxs, x_move, x_box), x_move)
        if num_moved == 0:
            np.testing.assert_array_equal(last_conf, x_move)
        assert num_moved <= 1, "More than one mol moved, something is wrong"
        last_conf = x_move
    # All moves are accepted, however the two waters could clash do to a proposal.
    assert accepted == moves


@pytest.mark.parametrize("moves", [1, 100])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("seed", [2023])
def test_bd_exchange_deterministic_moves(moves, precision, seed):
    """Given one water the exchange mover should accept every move and the results should be deterministic"""
    ff = Forcefield.load_default()
    system, conf, _, _ = builders.build_water_system(1.0, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), conf.shape[0])

    # Select a single mol
    group_idxs = all_group_idxs[:1]

    conf_idxs = np.array(group_idxs).reshape(-1)

    conf = conf[conf_idxs]

    box = np.eye(3) * 100.0

    N = conf.shape[0]

    params = nb.params[conf_idxs]

    cutoff = nb.potential.cutoff
    klass = custom_ops.BDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.BDExchangeMove_f64

    # Reference that makes a single proposal per move
    bdem_a = klass(N, group_idxs, params, DEFAULT_TEMP, nb.potential.beta, cutoff, seed, 1)
    # Test version that makes all proposals in a single move
    bdem_b = klass(N, group_idxs, params, DEFAULT_TEMP, nb.potential.beta, cutoff, seed, moves)

    iterative_moved_coords = conf.copy()
    for _ in range(moves):
        iterative_moved_coords, _ = bdem_a.move(iterative_moved_coords, box)
        assert not np.all(conf == iterative_moved_coords)  # We should move every time since its a single mol
    batch_moved_coords, _ = bdem_b.move(conf, box)
    # Moves should be deterministic regardless the number of steps taken per move
    np.testing.assert_array_equal(iterative_moved_coords, batch_moved_coords)


@pytest.mark.parametrize(
    "steps_per_move,moves",
    [(1, 2500), (10, 2500), (200000, 200000)],
)
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 5e-6, 5e-6), (np.float32, 1e-4, 2e-3)])
@pytest.mark.parametrize("seed", [2023])
def test_moves_in_a_water_box(steps_per_move, moves, precision, rtol, atol, seed):
    """Verify that the log acceptance probability between the reference and cuda implementation agree"""
    ff = Forcefield.load_default()
    system, conf, box, _ = builders.build_water_system(3.0, ff.water_ff)
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

    bdem = klass(N, group_idxs, params, DEFAULT_TEMP, nb.potential.beta, cutoff, seed, steps_per_move)

    ref_bdem = RefBDExchangeMove(nb.potential.beta, cutoff, params, group_idxs, DEFAULT_TEMP)

    assert bdem.last_log_probability() == 0.0, "First log probability expected to be zero"
    accepted = 0
    last_conf = conf
    for _ in range(moves // steps_per_move):
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
            np.testing.assert_allclose(image_frame(group_idxs, x_move, x_box), x_move)
            # Verify that the probabilities and per mol energies agree when we do accept moves
            # can only be done when we only attempt a single move per step
            if steps_per_move == 1:
                before_log_weights = ref_bdem.batch_log_weights(last_conf, box)
                after_log_weights, tested = ref_bdem.batch_log_weights_incremental(
                    last_conf, x_box, idx, new_pos, before_log_weights
                )
                np.testing.assert_array_equal(tested, x_move)
                np.testing.assert_allclose(
                    np.exp(bdem.last_log_probability()),
                    np.exp(np.minimum(logsumexp(before_log_weights) - logsumexp(after_log_weights), 0.0)),
                    rtol=rtol,
                    atol=atol,
                )
        if num_moved == 0:
            np.testing.assert_array_equal(last_conf, x_move)
        assert steps_per_move != 1 or num_moved <= 1, "More than one mol moved, something is wrong"
        last_conf = x_move
    assert bdem.n_proposed() == moves
    if moves < 10_000:
        assert accepted > 0, "No moves were made, nothing was tested"
    else:
        assert bdem.n_accepted() > 10
        np.testing.assert_allclose(0.0002, bdem.acceptance_fraction(), atol=5e-5)
    if steps_per_move == 1:
        np.testing.assert_allclose(bdem.acceptance_fraction(), accepted / moves)
        assert bdem.n_accepted() == accepted
    else:
        assert bdem.n_accepted() >= accepted
        assert bdem.acceptance_fraction() >= accepted / moves
