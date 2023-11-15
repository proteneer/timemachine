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
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.exchange.exchange_mover import TIBDExchangeMove as RefTIBDExchangeMove
from timemachine.md.exchange.exchange_mover import delta_r_np, get_water_groups
from timemachine.potentials import HarmonicBond, Nonbonded
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


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
    params = rng.random(size=(10, 4))

    ligand_idxs = np.array([0])
    radius = 1.0

    # Test group indices verification
    group_idxs = []
    with pytest.raises(RuntimeError, match="must provide at least one molecule"):
        klass(N, ligand_idxs, group_idxs, params, DEFAULT_TEMP, beta, cutoff, radius, seed, proposals_per_move)

    # Second molecule is not contiguous with first
    group_idxs = [[0, 1, 2], [4, 5]]
    with pytest.raises(RuntimeError, match="Molecules are not contiguous: mol 1"):
        klass(N, ligand_idxs, group_idxs, params, DEFAULT_TEMP, beta, cutoff, radius, seed, proposals_per_move)

    group_idxs = [[0, 1, 2], [3, 4]]
    with pytest.raises(RuntimeError, match="only support running with mols with constant size, got mixed sizes"):
        klass(N, ligand_idxs, group_idxs, params, DEFAULT_TEMP, beta, cutoff, radius, seed, proposals_per_move)

    group_idxs = [[]]
    with pytest.raises(RuntimeError, match="must provide non-empty molecule indices"):
        klass(N, ligand_idxs, group_idxs, params, DEFAULT_TEMP, beta, cutoff, radius, seed, proposals_per_move)

    # Proposals must be non-zero
    with pytest.raises(RuntimeError, match="proposals per move must be greater than 0"):
        klass(N, ligand_idxs, group_idxs, params, DEFAULT_TEMP, beta, cutoff, radius, seed, 0)

    # Radius must be non-zero
    with pytest.raises(RuntimeError, match="radius must be greater than 0.0"):
        klass(N, ligand_idxs, group_idxs, params, DEFAULT_TEMP, beta, cutoff, 0.0, seed, proposals_per_move)


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
        barostat=baro_impl,
    )
    ctxt.multiple_steps(1000)
    conf = ctxt.get_x_t()
    box = ctxt.get_box()
    return replace(initial_state, v0=ctxt.get_v_t(), x0=conf, box0=box)


@pytest.mark.parametrize("radius", [1.2])
@pytest.mark.parametrize(
    "steps_per_move,moves",
    [(1, 500), (5000, 5000)],
)
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 5e-6, 5e-6), (np.float32, 1e-4, 2e-3)])
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
                # TBD - FIX THIS TO BE MEANINGFUL
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
