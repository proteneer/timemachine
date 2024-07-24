from functools import partial
from importlib import resources
from typing import List, Optional
from unittest.mock import Mock, patch

import numpy as np
import pymbar
import pytest
from hypothesis import given, seed
from hypothesis.strategies import integers
from jax import grad, jacfwd, jacrev, value_and_grad
from scipy.optimize import check_grad, minimize

from timemachine.constants import DEFAULT_TEMP
from timemachine.fe import free_energy, topology, utils
from timemachine.fe.bar import ukln_to_ukn
from timemachine.fe.free_energy import (
    BarResult,
    HostConfig,
    HREXSimulationResult,
    IndeterminateEnergyWarning,
    MDParams,
    MinOverlapWarning,
    PairBarResult,
    Trajectory,
    batches,
    compute_potential_matrix,
    estimate_free_energy_bar,
    get_water_sampler_params,
    make_pair_bar_plots,
    run_sims_bisection,
    sample,
    trajectories_by_replica_to_by_state,
)
from timemachine.fe.rbfe import Host, setup_initial_state, setup_initial_states, setup_optimized_host
from timemachine.fe.single_topology import AtomMapFlags, SingleTopology
from timemachine.fe.stored_arrays import StoredArrays
from timemachine.fe.system import convert_omm_system
from timemachine.ff import Forcefield
from timemachine.md import builders
from timemachine.md.hrex import HREX, HREXDiagnostics
from timemachine.md.states import CoordsVelBox
from timemachine.potentials import BoundPotential, Nonbonded, SummedPotential
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


def assert_shapes_consistent(U, coords, sys_params, box):
    """assert U, grad(U) have the right shapes"""
    # can call U and get right shape
    energy = U(coords, sys_params, box)
    assert isinstance(energy, float)

    # can call grad(U) and get right shape
    du_dx, du_dp = grad(U, argnums=(0, 1))(coords, sys_params, box)
    assert du_dx.shape == coords.shape
    for p, p_prime in zip(sys_params, du_dp):
        assert p.shape == p_prime.shape


def assert_fwd_rev_consistent(f, x):
    """assert jacfwd(f)(x) == jacrev(f)(x)"""

    # forward-mode agrees with reverse-mode
    fwd_result = jacfwd(f)(x)
    rev_result = jacrev(f)(x)

    np.testing.assert_array_almost_equal(fwd_result, rev_result)


def assert_no_second_derivative(f, x):
    """assert an exception is raised if we try to access second derivatives of f"""

    # shouldn't be able to take second derivatives
    def div_f(x):
        return np.sum(grad(f)(x))

    problem = None
    try:
        # another grad should be a no-no
        grad(div_f)(x)
    except Exception as e:
        problem = e
    assert isinstance(problem, TypeError)


def assert_ff_optimizable(U, coords, sys_params, box, tol=1e-10):
    """define a differentiable loss function in terms of U, assert it can be minimized,
    and return initial params, optimized params, and the loss function"""

    nb_params = sys_params[-1]
    nb_params_shape = nb_params.shape

    def loss(nb_params):
        concat_params = sys_params[:-1] + [nb_params]
        return (U(coords, concat_params, box) - 0.1) ** 2

    x_0 = nb_params.flatten()

    def flat_loss(flat_nb_params):
        return loss(flat_nb_params.reshape(nb_params_shape))

    def fun(flat_nb_params):
        v, g = value_and_grad(flat_loss)(flat_nb_params)
        return float(v), np.array(g)

    # minimization successful
    result = minimize(fun, x_0, jac=True, tol=0)
    x_opt = result.x
    assert flat_loss(x_opt) < tol

    return x_0, x_opt, flat_loss


def test_functional():
    """Assert that
    * derivatives of U w.r.t. x and params accessible by grad(U) are of the correct shape,
    * a differentiable loss function in terms of U can be minimized,
    * forward-mode and reverse-mode differentiation of loss agree,
    * an exception is raised if we try to do something that requires second derivatives,
    * grad(nonlinear_function_in_terms_of_U) agrees with finite-difference,
    * requesting derivative w.r.t. box causes a runtime error
    """

    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, forcefield)

    vac_sys = st.setup_intermediate_state(0.5)
    x_a = utils.get_romol_conf(st.mol_a)
    x_b = utils.get_romol_conf(st.mol_b)
    coords = st.combine_confs(x_a, x_b)
    box = np.eye(3) * 100

    bps = vac_sys.get_U_fns()
    potentials = [bp.potential for bp in bps]
    sys_params = [np.array(bp.params) for bp in bps]

    for precision in [np.float32, np.float64]:
        U = SummedPotential(potentials, sys_params).to_gpu(precision).call_with_params_list

        # U, grad(U) have the right shapes
        assert_shapes_consistent(U, coords, sys_params, box)

        # can scipy.optimize a differentiable Jax function that calls U
        x_0, x_opt, flat_loss = assert_ff_optimizable(U, coords, sys_params, box)

        # jacfwd agrees with jacrev
        assert_fwd_rev_consistent(flat_loss, x_0)

        # calling grad twice shouldn't be allowed
        assert_no_second_derivative(flat_loss, x_0)

        # check grad by comparison to forward finite-difference
        if precision == np.float64:

            def low_dim_f(perturb: float) -> float:
                """low-dimensional input so that finite-difference isn't too expensive"""

                # scaling perturbation down by 1e-4 so that f(1.0) isn't 10^30ish...
                return flat_loss(x_opt + 1e-4 * perturb) ** 2

            perturbations = np.linspace(-1, 1, 10)

            for perturb in perturbations:
                abs_err = check_grad(low_dim_f, grad(low_dim_f), perturb, epsilon=1e-4)
                assert abs_err < 1.5e-3

        # grad w.r.t. box shouldn't be allowed
        with pytest.raises(RuntimeError) as e:
            _ = grad(U, argnums=2)(coords, sys_params, box)
        assert "box" in str(e).lower()


@pytest.mark.nocuda
def test_absolute_vacuum():
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols = utils.read_sdf(path_to_ligand)
    mol = mols[0]

    ff = Forcefield.load_default()
    ff_params = ff.get_params()

    bt = topology.BaseTopology(mol, ff)
    afe = free_energy.AbsoluteFreeEnergy(mol, bt)

    unbound_potentials, sys_params, masses = afe.prepare_vacuum_edge(ff_params)
    assert np.all(masses == utils.get_mol_masses(mol))
    np.testing.assert_array_almost_equal(afe.prepare_combined_coords(), utils.get_romol_conf(mol))


@pytest.mark.nocuda
def test_vacuum_and_solvent_edge_types():
    """Ensure that the values returned by the vacuum and solvent edges are all of the same type."""
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols = utils.read_sdf(path_to_ligand)
    mol = mols[0]

    ff = Forcefield.load_default()
    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(3.0, ff.water_ff, mols=[mol])
    host_system = HostConfig(solvent_system, solvent_coords, solvent_box, solvent_coords.shape[0])
    ff_params = ff.get_params()

    bt = topology.BaseTopology(mol, ff)
    afe = free_energy.AbsoluteFreeEnergy(mol, bt)

    vacuum_unbound_potentials, vacuum_sys_params, vacuum_masses = afe.prepare_vacuum_edge(ff_params)

    solvent_unbound_potentials, solvent_sys_params, solvent_masses = afe.prepare_host_edge(ff_params, host_system, 0.0)

    assert isinstance(vacuum_unbound_potentials, type(solvent_unbound_potentials))
    assert isinstance(vacuum_sys_params, type(solvent_sys_params))
    assert isinstance(vacuum_masses, type(solvent_masses))


@pytest.fixture(scope="module")
def hif2a_ligand_pair_single_topology():
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, forcefield)
    return st, forcefield


@pytest.fixture(scope="module")
def solvent_hif2a_ligand_pair_single_topology_lam0_state(hif2a_ligand_pair_single_topology):
    st, forcefield = hif2a_ligand_pair_single_topology
    solvent_sys, solvent_conf, solvent_box, solvent_top = builders.build_water_system(
        3.0, forcefield.water_ff, mols=[st.mol_a, st.mol_b]
    )
    solvent_host_config = HostConfig(solvent_sys, solvent_conf, solvent_box, solvent_conf.shape[0])
    solvent_host = setup_optimized_host(st, solvent_host_config)
    state = setup_initial_states(st, solvent_host, DEFAULT_TEMP, [0.0], 2023)[0]
    return state


@pytest.mark.parametrize("n_frames", [1, 10])
@pytest.mark.parametrize("local_steps", [0, 1])
@pytest.mark.parametrize("max_buffer_frames", [1])
def test_sample_max_buffer_frames_with_local_md(
    solvent_hif2a_ligand_pair_single_topology_lam0_state, n_frames, local_steps, max_buffer_frames
):
    """Ensure that if sample is called with max_buffer_frames combined with local MD it works. This failed previously
    due to trying to configure local md on the same context repeatedly. This was due to max_buffer_frames < n_frames which
    resulted in calling ctxt.setup_local_md multiple times.
    """
    steps_per_frame = 1
    n_eq_steps = 1

    md_params = MDParams(n_frames, n_eq_steps, steps_per_frame, 2023, local_steps=local_steps)
    traj = sample(solvent_hif2a_ligand_pair_single_topology_lam0_state, md_params, max_buffer_frames)
    assert isinstance(traj.frames, StoredArrays)
    assert len(traj.frames) == n_frames


@pytest.mark.nocuda
@given(integers(min_value=1))
@seed(2023)
def test_batches_of_nothing(batch_size):
    assert list(batches(0, batch_size)) == []


@pytest.mark.nocuda
@given(integers(min_value=1, max_value=1000), integers(min_value=1))
@seed(2023)
def test_batches(n, batch_size):
    assert sum(batches(n, batch_size)) == n
    assert all(batch == batch_size for batch in list(batches(n, batch_size))[:-1])
    *_, last = batches(n, batch_size)
    assert 0 < last <= batch_size


@pytest.fixture(scope="module")
def hif2a_ligand_pair_single_topology_lam0_state():
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, forcefield)
    state = setup_initial_state(st, 0.0, None, DEFAULT_TEMP, 2023)
    return state


@pytest.mark.parametrize("seed", [2024])
@pytest.mark.parametrize("host_name", [None, "solvent", "complex"])
def test_initial_state_interacting_ligand_atoms(host_name, seed):
    lambdas = np.linspace(0.0, 1.0, 4)
    forcefield = Forcefield.load_default()
    host_config: Optional[HostConfig] = None
    host: Optional[Host] = None

    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    if host_name == "complex":
        with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as protein_path:
            host_sys, host_conf, box, _, num_water_atoms = builders.build_protein_system(
                str(protein_path), forcefield.protein_ff, forcefield.water_ff, mols=[mol_a, mol_b]
            )
            box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
        host_config = HostConfig(host_sys, host_conf, box, num_water_atoms)
    elif host_name == "solvent":
        host_sys, host_conf, box, _ = builders.build_water_system(4.0, forcefield.water_ff, mols=[mol_a, mol_b])
        box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
        host_config = HostConfig(host_sys, host_conf, box, host_conf.shape[0])

    single_topology = SingleTopology(mol_a, mol_b, core, forcefield)

    host_atoms = 0
    if host_config is not None:
        # system, masses = convert_omm_system(host_config.omm_system)
        host = setup_optimized_host(single_topology, host_config)
        host_atoms += len(host_conf)

    initial_states = setup_initial_states(
        single_topology, host, DEFAULT_TEMP, lambdas, seed=seed, min_cutoff=0.7 if host_name is not None else None
    )

    for state in initial_states:
        mol_a_atoms = state.ligand_idxs[single_topology.c_flags != AtomMapFlags.MOL_B]
        mol_b_atoms = state.ligand_idxs[single_topology.c_flags != AtomMapFlags.MOL_A]
        core_atoms = state.ligand_idxs[single_topology.c_flags == AtomMapFlags.CORE]
        assert state.interacting_atoms is not None
        if state.lamb == 0.0:
            assert set(state.interacting_atoms) == set(mol_a_atoms)
        elif state.lamb == 1.0:
            assert set(state.interacting_atoms) == set(mol_b_atoms)
        else:
            assert set(state.interacting_atoms) == set(core_atoms)


@pytest.mark.nocuda
@patch("timemachine.fe.free_energy.plot_overlap_detail_figure")
def test_plot_pair_bar_plots(mock_fig, hif2a_ligand_pair_single_topology_lam0_state):
    pair_result = PairBarResult(
        [hif2a_ligand_pair_single_topology_lam0_state] * 2,
        [
            BarResult(
                0.0,
                0.0,
                np.zeros(4),
                0.0,
                np.zeros(4),
                np.zeros((4, 2, 2, 1)),
            )
        ],
    )
    make_pair_bar_plots(pair_result, DEFAULT_TEMP, "")
    assert mock_fig.call_args is not None
    expected_potentials = {
        "HarmonicBond",
        "HarmonicAngleStable",
        "PeriodicTorsion",
        "NonbondedPairListPrecomputed",
        "ChiralAtomRestraint",
    }
    assert set(mock_fig.call_args.args[0]) == expected_potentials


@pytest.mark.nocuda
@pytest.mark.parametrize("num_windows", [2, 5])
def test_get_water_sampler_params(num_windows):
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    # Use the simple charges simply because it is faster
    forcefield = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, forcefield)
    solvent_sys, solvent_conf, solvent_box, solvent_top = builders.build_water_system(
        3.0, forcefield.water_ff, mols=[mol_a, mol_b]
    )
    num_host_atoms = solvent_conf.shape[0]
    host_system, masses = convert_omm_system(solvent_sys)
    solvent_host = Host(host_system, masses, solvent_conf, solvent_box, num_host_atoms)
    mol_a_only_atoms = np.array([i for i in range(st.get_num_atoms()) if st.c_flags[i] == 1])
    mol_b_only_atoms = np.array([i for i in range(st.get_num_atoms()) if st.c_flags[i] == 2])
    for lamb in np.linspace(0.0, 1.0, num_windows, endpoint=True):
        state = setup_initial_state(st, lamb, solvent_host, DEFAULT_TEMP, 2024)
        water_sampler_nb_params = get_water_sampler_params(state)
        nb_pot = next(p.potential for p in state.potentials if isinstance(p.potential, Nonbonded))
        ligand_water_params = st._get_guest_params(
            forcefield.q_handle_solv, forcefield.lj_handle_solv, lamb, nb_pot.cutoff
        )
        if lamb == 0.0:
            assert np.all(ligand_water_params[mol_a_only_atoms][:, 3] == 0.0)
            assert np.all(ligand_water_params[mol_b_only_atoms][:, 3] == nb_pot.cutoff)
        elif lamb == 1.0:
            assert np.all(ligand_water_params[mol_a_only_atoms][:, 3] == nb_pot.cutoff)
            assert np.all(ligand_water_params[mol_b_only_atoms][:, 3] == 0.0)

        np.testing.assert_array_equal(water_sampler_nb_params[num_host_atoms:], ligand_water_params)


def test_run_sims_bisection_early_stopping(hif2a_ligand_pair_single_topology_lam0_state):
    initial_state = hif2a_ligand_pair_single_topology_lam0_state

    def make_initial_state(_: float):
        return initial_state

    md_params = MDParams(1, 1, 1, 2023)

    n_bisections = 3

    run_sims_bisection_early_stopping = partial(
        run_sims_bisection,
        [0.0, 1.0],
        make_initial_state,
        md_params,
        n_bisections=n_bisections,
        temperature=DEFAULT_TEMP,
        verbose=False,
    )

    # runs all n_bisections iterations by default
    results = run_sims_bisection_early_stopping()[0]
    assert len(results) == 1 + n_bisections  # initial result + bisection iterations

    def result_with_overlap(overlap):
        return BarResult(np.nan, np.nan, np.empty, overlap, np.empty, np.empty)

    with patch("timemachine.fe.free_energy.estimate_free_energy_bar") as mock_estimate_free_energy_bar:
        # overlap does not improve; should run all n_bisections iterations
        mock_estimate_free_energy_bar.return_value = result_with_overlap(0.0)
        with pytest.warns(MinOverlapWarning):
            results = run_sims_bisection_early_stopping(min_overlap=0.4)[0]
        assert len(results) == 1 + n_bisections

        # min_overlap satisfied by initial states
        mock_estimate_free_energy_bar.return_value = result_with_overlap(0.5)
        results = run_sims_bisection_early_stopping(min_overlap=0.4)[0]
        assert len(results) == 1

        # min_overlap achieved after 1 iteration
        mock_estimate_free_energy_bar.side_effect = [result_with_overlap(overlap) for overlap in [0.0, 0.5, 0.5]]
        results = run_sims_bisection_early_stopping(min_overlap=0.4)[0]
        assert len(results) == 1 + 1

        # min_overlap achieved after 2 iterations
        mock_estimate_free_energy_bar.side_effect = [
            result_with_overlap(overlap) for overlap in [0.0, 0.5, 0.3, 0.5, 0.6]
        ]
        results = run_sims_bisection_early_stopping(min_overlap=0.4)[0]
        assert len(results) == 1 + 2


@pytest.mark.nocuda
def test_estimate_free_energy_bar_with_energy_overflow():
    """Ensure that we handle NaNs in u_kln inputs (e.g. due to overflow in potential evaluation)."""
    rng = np.random.default_rng(2023)
    u_kln = rng.uniform(-1, 1, (2, 2, 100))

    _ = estimate_free_energy_bar(np.array([u_kln]), DEFAULT_TEMP)

    u_kln_with_nan = np.array(u_kln)
    u_kln_with_nan[0, 1, 10] = np.nan

    # pymbar.MBAR fails with LinAlgError
    with pytest.raises(SystemError, match="LinAlgError"):
        u_kn, N_k = ukln_to_ukn(u_kln_with_nan)
        _ = pymbar.MBAR(u_kn, N_k)

    # should return finite results with warning
    with pytest.warns(IndeterminateEnergyWarning, match="NaN"):
        result_with_nan = estimate_free_energy_bar(np.array([u_kln_with_nan]), DEFAULT_TEMP)

    assert np.isfinite(result_with_nan.dG)
    assert np.isfinite(result_with_nan.dG_err)

    u_kln_with_inf = np.array(u_kln)
    u_kln_with_inf[0, 1, 10] = np.inf

    # should give the same result with inf
    result_with_inf = estimate_free_energy_bar(np.array([u_kln_with_inf]), DEFAULT_TEMP)
    assert result_with_nan.dG == result_with_inf.dG
    assert result_with_nan.dG_err == result_with_inf.dG_err
    np.testing.assert_array_equal(result_with_nan.dG_err_by_component, result_with_inf.dG_err_by_component)
    np.testing.assert_array_equal(result_with_nan.overlap, result_with_inf.overlap)


@pytest.mark.parametrize(
    "n_states,max_delta_states", [(1, 1), (1, None), (1, 2), (2, 1), (3, 6), (6, 3), (6, None), (30, 5)]
)
@pytest.mark.parametrize("seed", [2024, 2025])
def test_compute_potential_matrix(hif2a_ligand_pair_single_topology, n_states: int, max_delta_states: int | None, seed):
    st, _ = hif2a_ligand_pair_single_topology
    states = [st.setup_intermediate_state(lam) for lam in np.linspace(0.0, 1.0, n_states)]

    def make_summed_potential(bps: List[BoundPotential]):
        potentials = [bp.potential for bp in bps]
        params = [bp.params for bp in bps]
        return SummedPotential(potentials, params).bind_params_list(params)

    bps = [make_summed_potential(s.get_U_fns()) for s in states]
    potential = bps[0].potential
    params_by_state = np.array([bp.params for bp in bps])

    conf_a = utils.get_romol_conf(st.mol_a)
    conf_b = utils.get_romol_conf(st.mol_b)
    conf = st.combine_confs(conf_a, conf_b)

    rng = np.random.default_rng(seed)
    confs = conf + rng.normal(0.0, 0.01, (len(states), st.get_num_atoms(), 3))
    boxes = 100.0 * np.eye(3) + rng.uniform(-1.0, 1.0, (len(states), 3, 3))
    xvbs = [CoordsVelBox(conf, np.zeros_like(conf), box) for conf, box in zip(confs, boxes)]

    unbound_impl = potential.to_gpu(np.float32).unbound_impl

    _, _, U_ref = unbound_impl.execute_batch(confs, params_by_state, boxes, False, False, True)
    assert np.all(np.isfinite(U_ref))

    replica_idx_by_state = rng.choice(n_states, size=n_states, replace=False).tolist()
    hrex = HREX(xvbs, replica_idx_by_state)

    U_test = compute_potential_matrix(unbound_impl, hrex, params_by_state, max_delta_states)

    state_idx = np.arange(n_states)
    state_idx_by_replica = np.argsort(replica_idx_by_state)
    is_computed = (
        np.full((n_states, n_states), True)
        if max_delta_states is None
        else np.abs(state_idx_by_replica[:, None] - state_idx[None, :]) <= max_delta_states
    )
    np.testing.assert_array_equal(U_ref[is_computed], U_test[is_computed])
    assert np.all(np.isinf(U_test[~is_computed]))


@pytest.mark.nogpu
@pytest.mark.parametrize("seed", [2024])
@pytest.mark.parametrize("n_states, n_iters", [(2, 2), (5, 10), (24, 100)])
@pytest.mark.parametrize("n_atoms", [2182])
def test_trajectories_by_replica_to_by_state(seed, n_states, n_iters, n_atoms):
    rng = np.random.default_rng(seed)
    frames = rng.uniform(size=(n_states, n_iters, n_atoms, 3))

    atom_idxs = np.arange(rng.integers(n_atoms))

    states = np.arange(n_states)
    replica_idx_by_state_by_iter = [rng.choice(states, size=(n_states), replace=False).tolist() for _ in range(n_iters)]

    dummy_box = np.eye(3) * 100.0
    trajs = []
    for state_frames in frames:
        stored_frames = StoredArrays.from_chunks([state_frames])
        traj = Trajectory(
            frames=stored_frames,
            boxes=[dummy_box] * len(stored_frames),
            final_velocities=None,
            final_barostat_volume_scale_factor=None,
        )
        trajs.append(traj)

    sim_res = HREXSimulationResult(
        final_result=Mock(),
        plots=Mock(),
        hrex_plots=Mock(),
        trajectories=trajs,
        md_params=Mock(),
        intermediate_results=[Mock()],
        hrex_diagnostics=HREXDiagnostics(replica_idx_by_state_by_iter, []),
    )
    traj_by_replica = sim_res.extract_trajectories_by_replica(atom_idxs)
    traj_by_state = trajectories_by_replica_to_by_state(traj_by_replica, replica_idx_by_state_by_iter)
    np.testing.assert_array_equal(traj_by_state, frames[:, :, atom_idxs])
