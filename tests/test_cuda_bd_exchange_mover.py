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
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops
from timemachine.md import builders
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.exchange.exchange_mover import BDExchangeMove as RefBDExchangeMove
from timemachine.potentials import HarmonicBond, Nonbonded
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


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
    bdem = klass(N, group_idxs, params, DEFAULT_TEMP, nb.potential.beta, cutoff, seed, proposals_per_move, 1)

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
            ref_log_prob = np.minimum(logsumexp(before_log_weights) - logsumexp(after_log_weights), 0.0)
            ref_prob = np.exp(ref_log_prob)
            assert np.isfinite(ref_prob) and ref_prob > 0.0
            np.testing.assert_allclose(
                np.exp(bdem.last_log_probability()),
                ref_prob,
                rtol=rtol,
                atol=atol,
            )
            # The molecules should all be imaged in the home box
            np.testing.assert_allclose(image_frame(group_idxs, x_move, x_box), x_move)
        elif num_moved == 0:
            np.testing.assert_array_equal(last_conf, x_move)
        assert num_moved <= 1, "More than one mol moved, something is wrong"
        last_conf = x_move
    # All moves are accepted, however the two waters could clash do to a proposal.
    assert accepted == moves


@pytest.mark.parametrize("precision", [np.float32])
@pytest.mark.parametrize("seed", [2023])
def test_bias_deletion_bulk_water_with_context(precision, seed):
    ff = Forcefield.load_default()
    system, conf, box, _ = builders.build_water_system(4.0, ff.water_ff)
    bps, masses = openmm_deserializer.deserialize_system(system, cutoff=1.2)
    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    bond_list = get_bond_list(bond_pot)
    all_group_idxs = get_group_indices(bond_list, conf.shape[0])

    # only act on waters
    water_idxs = [group for group in all_group_idxs if len(group) == 3]

    dt = 1.5e-3

    bound_impls = []

    for potential in bps:
        bound_impls.append(potential.to_gpu(precision=np.float32).bound_impl)

    klass = custom_ops.BDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.BDExchangeMove_f64

    proposals_per_move = 1000
    interval = 100
    steps = 500
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
    ctxt.multiple_steps(steps)
    assert bdem.n_proposed() == (steps // interval) * proposals_per_move
    assert bdem.n_accepted() > 0


@pytest.mark.parametrize("proposals_per_move", [1, 100])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("seed", [2023])
def test_bd_exchange_deterministic_moves(proposals_per_move, precision, seed):
    """Given one water the exchange mover should accept every move and the results should be deterministic given the same seed and number of proposals per move


    There are three forms of determinism we require:
    * Constructing an exchange move produces the same results every time
    * Calling an exchange move with one proposals per move or K proposals per move produce the same state.
      * It is difficult to test each move when there are K proposals per move so we need to know that it matches the single proposals per move case
    * TBD: When we attempt K proposals in a batch (each proposal is made up of K proposals) it produces the same as the serial version
    """
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
    bdem_a = klass(N, group_idxs, params, DEFAULT_TEMP, nb.potential.beta, cutoff, seed, 1, 1)
    # Test version that makes all proposals in a single move
    bdem_b = klass(N, group_idxs, params, DEFAULT_TEMP, nb.potential.beta, cutoff, seed, proposals_per_move, 1)

    iterative_moved_coords = conf.copy()
    for _ in range(proposals_per_move):
        iterative_moved_coords, _ = bdem_a.move(iterative_moved_coords, box)
        assert not np.all(conf == iterative_moved_coords)  # We should move every time since its a single mol
    batch_moved_coords, _ = bdem_b.move(conf, box)
    # Moves should be deterministic regardless the number of steps taken per move
    np.testing.assert_array_equal(iterative_moved_coords, batch_moved_coords)

    assert bdem_a.n_accepted() > 0
    assert bdem_a.n_proposed() == proposals_per_move
    assert bdem_a.n_accepted() == bdem_b.n_accepted()
    assert bdem_a.n_proposed() == bdem_b.n_proposed()


@pytest.mark.parametrize(
    "steps_per_move,moves,box_size",
    [
        (1, 2500, 3.0),
        (2500, 2500, 3.0),
        (250000, 250000, 3.0),
        # The 6.0nm box triggers a failure that would occur with systems of certain sizes, may be flaky in identifying issues
        pytest.param(1, 2500, 6.0, marks=pytest.mark.nightly(reason="slow")),
    ],
)
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 5e-6, 5e-6), (np.float32, 1e-4, 2e-3)])
@pytest.mark.parametrize("seed", [2023])
def test_moves_in_a_water_box(steps_per_move, moves, box_size, precision, rtol, atol, seed):
    """Verify that the log acceptance probability between the reference and cuda implementation agree"""
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

    bdem = klass(N, group_idxs, params, DEFAULT_TEMP, nb.potential.beta, cutoff, seed, steps_per_move, 1)

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
                ref_log_prob = np.minimum(logsumexp(before_log_weights) - logsumexp(after_log_weights), 0.0)
                ref_prob = np.exp(ref_log_prob)
                assert np.isfinite(ref_prob) and ref_prob > 0.0
                np.testing.assert_allclose(
                    np.exp(bdem.last_log_probability()),
                    ref_prob,
                    rtol=rtol,
                    atol=atol,
                )
        elif num_moved == 0:
            np.testing.assert_array_equal(last_conf, x_move)
        assert steps_per_move != 1 or num_moved <= 1, "More than one mol moved, something is wrong"
        last_conf = x_move
    assert bdem.n_proposed() == moves
    if moves < 10_000:
        assert accepted > 0, "No moves were made, nothing was tested"
    else:
        assert bdem.n_accepted() > 10
        assert bdem.acceptance_fraction() >= 0.0001
    if steps_per_move == 1:
        np.testing.assert_allclose(bdem.acceptance_fraction(), accepted / moves)
        assert bdem.n_accepted() == accepted
    else:
        assert bdem.n_accepted() >= accepted
        assert bdem.acceptance_fraction() >= accepted / moves


@pytest.fixture(scope="module")
def hif2a_complex():
    seed = 2023
    ff = Forcefield.load_default()
    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_ligand:
        complex_system, conf, box, _, _ = builders.build_protein_system(str(path_to_ligand), ff.protein_ff, ff.water_ff)
    bps, masses = openmm_deserializer.deserialize_system(complex_system, cutoff=1.2)
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    bond_list = get_bond_list(bond_pot)
    all_group_idxs = get_group_indices(bond_list, conf.shape[0])

    # Equilibrate the system a bit before hand, which reduces clashes in the system which results greater differences
    # between the reference and test case.
    dt = 1.5e-3
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
    ctxt.multiple_steps(1000)
    conf = ctxt.get_x_t()
    box = ctxt.get_box()
    return complex_system, conf, box


@pytest.mark.parametrize(
    "steps_per_move,moves",
    [(1, 500), (5000, 5000)],
)
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 5e-6, 5e-6), (np.float32, 1e-4, 2e-3)])
@pytest.mark.parametrize("seed", [2023])
def test_moves_with_complex(hif2a_complex, steps_per_move, moves, precision, rtol, atol, seed):
    complex_system, conf, box = hif2a_complex
    bps, masses = openmm_deserializer.deserialize_system(complex_system, cutoff=1.2)
    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    bond_list = get_bond_list(bond_pot)
    all_group_idxs = get_group_indices(bond_list, conf.shape[0])

    # only act on waters
    water_idxs = [group for group in all_group_idxs if len(group) == 3]

    # Re-image coords so that everything is imaged to begin with
    conf = image_frame(all_group_idxs, conf, box)

    N = conf.shape[0]

    params = nb.params

    cutoff = nb.potential.cutoff
    klass = custom_ops.BDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.BDExchangeMove_f64

    bdem = klass(N, water_idxs, params, DEFAULT_TEMP, nb.potential.beta, cutoff, seed, steps_per_move, 1)

    ref_bdem = RefBDExchangeMove(nb.potential.beta, cutoff, params, water_idxs, DEFAULT_TEMP)

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
                before_log_weights = ref_bdem.batch_log_weights(last_conf, box)
                after_log_weights, tested = ref_bdem.batch_log_weights_incremental(
                    last_conf, x_box, idx, new_pos, before_log_weights
                )
                np.testing.assert_array_equal(tested, x_move)
                ref_log_prob = np.minimum(logsumexp(before_log_weights) - logsumexp(after_log_weights), 0.0)
                ref_prob = np.exp(ref_log_prob)
                assert np.isfinite(ref_prob) and ref_prob > 0.0
                np.testing.assert_allclose(
                    np.exp(bdem.last_log_probability()),
                    ref_prob,
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


@pytest.mark.parametrize(
    "steps_per_move,moves",
    [pytest.param(1, 15000, marks=pytest.mark.nightly(reason="slow")), (15000, 15000)],
)
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 5e-6, 5e-6), (np.float32, 1e-4, 2e-3)])
@pytest.mark.parametrize("seed", [2023])
def test_moves_with_complex_and_ligand(hif2a_rbfe_state, steps_per_move, moves, precision, rtol, atol, seed):
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
    klass = custom_ops.BDExchangeMove_f32
    if precision == np.float64:
        klass = custom_ops.BDExchangeMove_f64

    bdem = klass(N, water_idxs, params, DEFAULT_TEMP, nb.potential.beta, cutoff, seed, steps_per_move, 1)

    ref_bdem = RefBDExchangeMove(nb.potential.beta, cutoff, params, water_idxs, DEFAULT_TEMP)

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
                before_log_weights = ref_bdem.batch_log_weights(last_conf, box)
                after_log_weights, tested = ref_bdem.batch_log_weights_incremental(
                    last_conf, x_box, idx, new_pos, before_log_weights
                )
                np.testing.assert_array_equal(tested, x_move)
                ref_log_prob = np.minimum(logsumexp(before_log_weights) - logsumexp(after_log_weights), 0.0)
                ref_prob = np.exp(ref_log_prob)
                assert np.isfinite(ref_prob) and ref_prob > 0.0
                np.testing.assert_allclose(
                    np.exp(bdem.last_log_probability()),
                    ref_prob,
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
