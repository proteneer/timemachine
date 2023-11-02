import numpy as np
import pytest
from scipy.special import logsumexp

from timemachine.constants import DEFAULT_TEMP
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import custom_ops
from timemachine.md import builders
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.exchange.exchange_mover import BDExchangeMove as RefBDExchangeMove
from timemachine.potentials import HarmonicBond, Nonbonded


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
    group_idxs = all_group_idxs[:5]

    conf_idxs = np.array(group_idxs).reshape(-1)

    conf = conf[conf_idxs]
    # Set the two waters ontop of each other
    conf[group_idxs[0], :] = conf[group_idxs[1], :]

    box = np.eye(3) * 100.0

    N = conf.shape[0]

    params = nb.params[conf_idxs]

    cutoff = nb.potential.cutoff
    klass = custom_ops.BDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.BDExchangeMove_f64

    bdem = klass(N, group_idxs, params, DEFAULT_TEMP, nb.potential.beta, cutoff, seed)

    ref_bdem = RefBDExchangeMove(nb.potential.beta, cutoff, params, group_idxs, DEFAULT_TEMP)

    assert bdem.last_log_probability() == 0.0, "First log probability expected to be zero"
    num_steps = 1
    last_conf = conf
    accepted = 0
    for _ in range(moves):
        before_log_weights = ref_bdem.batch_log_weights(last_conf, box)
        x_move, x_box = bdem.move(last_conf, box, num_steps)
        after_log_weights = ref_bdem.batch_log_weights(x_move, x_box)
        # The box will never change
        np.testing.assert_array_equal(box, x_box)
        num_moved = 0
        for mol_idxs in group_idxs:
            if not np.all(x_move[mol_idxs] == last_conf[mol_idxs]):
                num_moved += 1
        if num_moved > 0:
            accepted += 1
            # Verify that the probabilities agree when we do accept moves
            np.testing.assert_allclose(
                np.exp(bdem.last_log_probability()),
                np.exp(np.minimum(logsumexp(before_log_weights) - logsumexp(after_log_weights), 0.0)),
                rtol=rtol,
                atol=atol,
            )
        if num_moved == 0:
            np.testing.assert_array_equal(last_conf, x_move)
        assert num_moved <= 1, "More than one mol moved, something is wrong"
        last_conf = x_move
    assert accepted > 0


@pytest.mark.parametrize("moves", [1, 100])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("seed", [2023])
def test_bd_exchange_deterministic_moves(moves, precision, seed):
    """Given two waters directly on top of each other in a box, the exchange mover should accept almost any move"""
    ff = Forcefield.load_default()
    system, conf, _, _ = builders.build_water_system(1.0, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), conf.shape[0])

    # Select a single mol
    group_idxs = all_group_idxs[:2]

    conf_idxs = np.array(group_idxs).reshape(-1)

    conf = conf[conf_idxs]

    box = np.eye(3) * 100.0

    N = conf.shape[0]

    params = nb.params[conf_idxs]

    cutoff = nb.potential.cutoff
    klass = custom_ops.BDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.BDExchangeMove_f64

    bdem_a = klass(N, group_idxs, params, DEFAULT_TEMP, nb.potential.beta, cutoff, seed)
    bdem_b = klass(N, group_idxs, params, DEFAULT_TEMP, nb.potential.beta, cutoff, seed)

    iterative_moved_coords = conf.copy()
    for _ in range(moves):
        iterative_moved_coords, _ = bdem_a.move(iterative_moved_coords, box, 1)
        assert not np.all(conf == iterative_moved_coords)  # We should move every time since its a single mol
    batch_moved_coords, _ = bdem_b.move(conf, box, moves)
    # Moves should be deterministic regardless the number of steps taken per move
    np.testing.assert_array_equal(iterative_moved_coords, batch_moved_coords)


@pytest.mark.parametrize("steps_per_move", [1, 10])
@pytest.mark.parametrize("moves", [2500])
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

    params = nb.params

    cutoff = nb.potential.cutoff
    klass = custom_ops.BDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.BDExchangeMove_f64

    bdem = klass(N, group_idxs, params, DEFAULT_TEMP, nb.potential.beta, cutoff, seed)

    ref_bdem = RefBDExchangeMove(nb.potential.beta, cutoff, params, group_idxs, DEFAULT_TEMP)

    assert bdem.last_log_probability() == 0.0, "First log probability expected to be zero"
    accepted = 0
    last_conf = conf
    for _ in range(moves // steps_per_move):
        x_move, x_box = bdem.move(last_conf, box, steps_per_move)
        # The box will never change
        np.testing.assert_array_equal(box, x_box)
        num_moved = 0
        for i, mol_idxs in enumerate(group_idxs):
            if not np.all(x_move[mol_idxs] == last_conf[mol_idxs]):
                num_moved += 1
        if num_moved > 0:
            accepted += 1
            # Verify that the probabilities and per mol energies agree when we do accept moves
            # can only be done when we only attempt a single move per step
            if steps_per_move == 1:
                before_log_weights = ref_bdem.batch_log_weights(last_conf, box)
                after_log_weights = ref_bdem.batch_log_weights(x_move, x_box)
                np.testing.assert_allclose(
                    np.exp(bdem.last_log_probability()),
                    np.exp(np.minimum(logsumexp(before_log_weights) - logsumexp(after_log_weights), 0.0)),
                    rtol=rtol,
                    atol=atol,
                )
        if num_moved == 0:
            np.testing.assert_array_equal(last_conf, x_move)
        assert num_moved <= 1, "More than one mol moved, something is wrong"
        last_conf = x_move
    assert accepted > 0, "No moves were made, nothing was tested"
    if steps_per_move == 1:
        assert bdem.n_accepted() == accepted
    else:
        assert bdem.n_accepted() >= accepted
    assert bdem.n_proposed() == moves
    np.testing.assert_allclose(bdem.acceptance_fraction(), accepted / moves)
