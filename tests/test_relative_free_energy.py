# test that we can run relative free energy simulations in complex and in solvent
# this doesn't test for accuracy, just that everything mechanically runs.
from unittest.mock import Mock, patch
from warnings import catch_warnings

import numpy as np
import pytest

from timemachine.fe.free_energy import (
    HREXParams,
    HREXSimulationResult,
    MDParams,
    PairBarResult,
    SimulationResult,
    image_frames,
    sample,
)
from timemachine.fe.rbfe import (
    estimate_relative_free_energy,
    estimate_relative_free_energy_bisection,
    estimate_relative_free_energy_bisection_hrex,
    rebalance_lambda_schedule,
    run_solvent,
    run_vacuum,
)
from timemachine.ff import Forcefield
from timemachine.md import builders
from timemachine.md.barostat.utils import compute_box_center
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology
from timemachine.utils import path_to_internal_file


def run_bitwise_reproducibility(mol_a, mol_b, core, forcefield, md_params, estimate_relative_free_energy_fn):
    # test that we can bitwise reproduce our trajectory using the initial state information

    box_width = 4.0
    n_windows = 3
    solvent_host_config = builders.build_water_system(box_width, forcefield.water_ff, mols=[mol_a, mol_b])
    solvent_host_config.box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes

    solvent_res = estimate_relative_free_energy_fn(
        mol_a,
        mol_b,
        core,
        forcefield,
        solvent_host_config,
        md_params=md_params,
        prefix="solvent",
        lambda_interval=(0.01, 0.03),
        n_windows=n_windows,
    )

    all_frames, all_boxes = [], []
    for state in solvent_res.final_result.initial_states:
        traj = sample(state, solvent_res.md_params, max_buffer_frames=100)
        all_frames.append(traj.frames)
        all_boxes.append(traj.boxes)

    np.testing.assert_equal(solvent_res.frames, all_frames)
    np.testing.assert_equal(solvent_res.boxes, all_boxes)


def run_triple(mol_a, mol_b, core, forcefield, md_params: MDParams, protein_path, estimate_relative_free_energy_fn):
    lambda_interval = [0.01, 0.03]
    n_windows = 3

    def check_sim_result(sim_res: SimulationResult):
        assert len(sim_res.final_result.initial_states) == n_windows
        assert sim_res.final_result.initial_states[0].lamb == lambda_interval[0]
        assert sim_res.final_result.initial_states[-1].lamb == lambda_interval[1]
        assert sim_res.plots.dG_errs_png is not None
        assert sim_res.plots.overlap_summary_png is not None
        assert sim_res.plots.overlap_detail_png is not None

        assert len(sim_res.frames[0]) == md_params.n_frames
        assert len(sim_res.frames[-1]) == md_params.n_frames
        assert len(sim_res.boxes[0]) == md_params.n_frames
        assert len(sim_res.boxes[-1]) == md_params.n_frames
        assert sim_res.md_params == md_params

        if isinstance(sim_res, HREXSimulationResult):
            assert md_params.hrex_params

            assert len(sim_res.hrex_diagnostics.fraction_accepted_by_pair_by_iter) == md_params.n_frames
            assert all(len(fs) == n_windows - 1 for fs in sim_res.hrex_diagnostics.fraction_accepted_by_pair_by_iter)

            assert len(sim_res.hrex_diagnostics.replica_idx_by_state_by_iter) == md_params.n_frames
            assert all(len(fs) == n_windows for fs in sim_res.hrex_diagnostics.replica_idx_by_state_by_iter)

        def check_pair_bar_result(res: PairBarResult):
            n_pairs = len(res.initial_states) - 1
            assert len(res.bar_results) == n_pairs

            assert np.all(0.0 < np.asarray(res.dG_errs))
            assert np.linalg.norm(res.dG_errs) < 0.1

            assert res.overlap_by_component_by_lambda.shape[0] == n_pairs
            assert res.overlap_by_component_by_lambda.shape[1] == res.dG_err_by_component_by_lambda.shape[1]
            for overlaps in [res.overlaps, res.overlap_by_component_by_lambda]:
                assert np.all(0.0 < np.asarray(overlaps))
                assert np.all(np.asarray(overlaps) < 1.0 + 1e-5)  # epsilon to deal with loss of precision

            assert np.all(0.0 <= np.asarray(res.dG_err_by_component_by_lambda))
            assert np.linalg.norm(res.dG_err_by_component_by_lambda) < 0.1

        check_pair_bar_result(sim_res.final_result)
        for res in sim_res.intermediate_results:
            check_pair_bar_result(res)

    vacuum_res = estimate_relative_free_energy_fn(
        mol_a,
        mol_b,
        core,
        forcefield,
        host_config=None,
        md_params=md_params,
        prefix="vacuum",
        lambda_interval=lambda_interval,
        n_windows=n_windows,
    )
    print("vacuum")
    check_sim_result(vacuum_res)

    box_width = 4.0
    solvent_host_config = builders.build_water_system(box_width, forcefield.water_ff, mols=[mol_a, mol_b])
    solvent_host_config.box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    solvent_res = estimate_relative_free_energy_fn(
        mol_a,
        mol_b,
        core,
        forcefield,
        solvent_host_config,
        md_params=md_params,
        prefix="solvent",
        lambda_interval=lambda_interval,
        n_windows=n_windows,
    )

    print("solvent")
    check_sim_result(solvent_res)

    complex_host_config = builders.build_protein_system(protein_path, forcefield.protein_ff, forcefield.water_ff)
    complex_host_config.box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    complex_res = estimate_relative_free_energy_fn(
        mol_a,
        mol_b,
        core,
        forcefield,
        complex_host_config,
        md_params=md_params,
        prefix="complex",
        lambda_interval=lambda_interval,
        n_windows=n_windows,
    )

    print("complex")
    check_sim_result(complex_res)


@pytest.mark.nightly(reason="Slow!")
@pytest.mark.parametrize(
    "estimate_relative_free_energy_fn",
    [
        # estimate_relative_free_energy,
        # estimate_relative_free_energy_bisection,
        estimate_relative_free_energy_bisection_hrex,
    ],
)
def test_run_hif2a_test_system(estimate_relative_free_energy_fn):
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_default()

    md_params = MDParams(
        n_frames=100,
        n_eq_steps=1000,
        steps_per_frame=100,
        seed=2023,
        hrex_params=HREXParams(),
    )

    with path_to_internal_file("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as protein_path:
        run_triple(
            mol_a,
            mol_b,
            core,
            forcefield,
            md_params=md_params,
            protein_path=str(protein_path),
            estimate_relative_free_energy_fn=estimate_relative_free_energy_fn,
        )


@pytest.mark.nightly(reason="Slow!")
@pytest.mark.parametrize(
    "estimate_relative_free_energy_fn",
    [
        estimate_relative_free_energy,
        estimate_relative_free_energy_bisection,
        pytest.param(
            estimate_relative_free_energy_bisection_hrex,
            marks=pytest.mark.skip(
                reason="lambda window trajectories are not individually reproducible given InitialState due to mixing"
            ),
        ),
    ],
)
def test_run_hif2a_test_system_reproducibility(estimate_relative_free_energy_fn):
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_default()

    md_params = MDParams(
        n_frames=100,
        n_eq_steps=1000,
        steps_per_frame=100,
        seed=2023,
    )

    run_bitwise_reproducibility(
        mol_a,
        mol_b,
        core,
        forcefield,
        md_params=md_params,
        estimate_relative_free_energy_fn=estimate_relative_free_energy_fn,
    )


@pytest.mark.nogpu
def test_md_params_validation():
    frames = 5
    steps_per_frame = 2

    # assert steps_per_frame > 0
    with pytest.raises(AssertionError):
        MDParams(seed=2023, n_frames=frames, n_eq_steps=10, steps_per_frame=0)

    # assert n_frames > 0
    with pytest.raises(AssertionError):
        MDParams(seed=2023, n_frames=0, n_eq_steps=1, steps_per_frame=steps_per_frame)

    # assert that local steps <= steps per frame
    with pytest.raises(AssertionError):
        MDParams(seed=2023, n_frames=frames, n_eq_steps=10, local_steps=5, steps_per_frame=steps_per_frame)

    # assert that local steps >= 0
    with pytest.raises(AssertionError):
        MDParams(seed=2023, n_frames=frames, n_eq_steps=10, local_steps=-1, steps_per_frame=steps_per_frame)

    # assert that min_radius <= max_radius
    with pytest.raises(AssertionError):
        MDParams(
            seed=2023, n_frames=frames, n_eq_steps=10, min_radius=4.0, max_radius=1.0, steps_per_frame=steps_per_frame
        )

    # assert that min_radius >= 0.1
    with pytest.raises(AssertionError):
        MDParams(
            seed=2023, n_frames=frames, n_eq_steps=10, min_radius=0.09, max_radius=1.0, steps_per_frame=steps_per_frame
        )

    # assert that k >= 1.0
    with pytest.raises(AssertionError):
        MDParams(seed=2023, n_frames=frames, n_eq_steps=10, k=-1.0, steps_per_frame=steps_per_frame)

    # assert that k <= 1e6
    with pytest.raises(AssertionError):
        MDParams(seed=2023, n_frames=frames, n_eq_steps=10, k=1.0e7, steps_per_frame=steps_per_frame)


def test_am1bcc_vacuum():
    """Verify that AM1BCC forcefields can be used to run a relative vacuum simulation"""
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file("smirnoff_2_2_0_am1bcc.py")
    seed = 2024
    frames = 5
    windows = 2
    steps_per_frame = 5

    md_params = MDParams(n_frames=frames, n_eq_steps=10, seed=seed, steps_per_frame=steps_per_frame)

    res = run_vacuum(mol_a, mol_b, core, forcefield, None, md_params=md_params, n_windows=windows)

    assert len(res.frames[0]) == frames
    assert res.md_params == md_params


@pytest.mark.parametrize("freeze_reference", [True, False])
def test_local_md_parameters(freeze_reference):
    """Run RBFE methods with local steps mixed in"""

    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_default()
    seed = 2023
    frames = 5
    windows = 2
    steps_per_frame = 5

    md_params = MDParams(
        n_frames=frames,
        n_eq_steps=10,
        seed=seed,
        steps_per_frame=steps_per_frame,
        local_steps=steps_per_frame,
        min_radius=0.3,
        max_radius=1.0,
        freeze_reference=freeze_reference,
    )

    # Local MD not supported by vacuum, will reset local_steps to 0
    with catch_warnings(record=True) as w:
        res = run_vacuum(mol_a, mol_b, core, forcefield, None, md_params=md_params, n_windows=windows)
    # Several warnings raised here, look for a specific message
    assert "Vacuum simulations don't support local steps, will use all global steps" in [x.message.args[0] for x in w]

    assert len(res.frames[0]) == frames
    assert res.md_params != md_params
    assert res.md_params.local_steps == 0

    # All of the particles should have moved, since it was global MD
    assert np.all(res.frames[0][0] == res.frames[0][-1], axis=1).sum() == 0

    res, _ = run_solvent(
        mol_a,
        mol_b,
        core,
        forcefield,
        None,
        md_params=md_params,
        n_windows=windows,
    )
    assert len(res.frames[0]) == frames
    assert res.md_params == md_params

    # Some of the particles should have never moved
    assert np.all(res.frames[0][0] == res.frames[0][-1], axis=1).sum() > 0


def test_steps_per_frames():
    """Verifies that modifying steps per frames doesn't change result trajectory"""
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_default()
    seed = 2022
    frames = 5
    md_params = MDParams(n_frames=frames, n_eq_steps=10, steps_per_frame=2, seed=seed)
    res = run_vacuum(mol_a, mol_b, core, forcefield, None, md_params=md_params, n_windows=2)
    assert len(res.frames[0]) == frames

    frames = 2
    md_params = MDParams(n_frames=frames, n_eq_steps=10, steps_per_frame=5, seed=seed)
    test_res = run_vacuum(mol_a, mol_b, core, forcefield, None, md_params=md_params, n_windows=2)
    assert len(test_res.frames[0]) == frames
    assert len(test_res.frames) == 2
    # The last frame from the trajectories should match as num_frames * steps_per_frame are equal
    for frame, test_frame in zip(res.frames, test_res.frames):
        np.testing.assert_array_equal(frame[-1], test_frame[-1])


def test_imaging_frames():
    """Verify that imaging frames places ligand at center and all coordinates are close to being within the box.

    Does not check precision, as it is known to be lossy. Only to be used for post-processing/visualization."""
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_default()
    seed = 2022
    frames = 1
    steps_per_frame = 1
    equil_steps = 1

    md_params = MDParams(n_frames=frames, n_eq_steps=equil_steps, steps_per_frame=steps_per_frame, seed=seed)

    windows = 2
    res, _ = run_solvent(
        mol_a,
        mol_b,
        core,
        forcefield,
        None,
        md_params=md_params,
        n_windows=windows,
    )

    # A buffer, as imaging doesn't ensure everything is perfectly in the box
    padding = 0.3

    for i, (frames, boxes) in enumerate(zip(res.frames, res.boxes)):
        initial_state = res.final_result.initial_states[i]
        box_center = compute_box_center(boxes[0])
        box_extents = np.max(boxes, axis=(0, 1))

        # Verify that coordinates are either outside of the box or below zero
        assert np.any(np.max(frames, axis=(0, 1)) > box_extents + padding) or np.any(
            np.min(frames, axis=(0, 1)) < -padding
        )
        # Ligand won't be near center of box
        assert not np.allclose(np.mean(frames[0][initial_state.ligand_idxs], axis=0), box_center)

        imaged = image_frames(initial_state, frames, boxes)

        # Verify that after imaged, coordinates are within padding of the box extents
        assert np.all(np.max(imaged, axis=(0, 1)) <= box_extents + padding) and np.all(
            np.min(imaged, axis=(0, 1)) >= -padding
        )
        # Verify that ligand was centered in the box
        np.testing.assert_allclose(np.mean(imaged[0][initial_state.ligand_idxs], axis=0), box_center)


@pytest.mark.parametrize(
    "estimate_relative_free_energy_fn",
    [estimate_relative_free_energy, estimate_relative_free_energy_bisection],
)
def test_rbfe_with_1_window(estimate_relative_free_energy_fn):
    """Should not be able to run a relative free energy calculation with a single window"""
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_default()
    seed = 2022
    md_params = MDParams(n_frames=1, n_eq_steps=10, steps_per_frame=1, seed=seed)
    with pytest.raises(AssertionError):
        estimate_relative_free_energy_fn(
            mol_a,
            mol_b,
            core,
            forcefield,
            None,
            md_params=md_params,
            prefix="failure",
            n_windows=1,
        )


def test_rbfe_fallback_from_near_zero_overlap():
    """Should fall back from [bisection, rebalancing] to [bisection] alone
    (with a UserWarning mentioning "overlap")
    when output of bisection has insufficient overlap"""
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_default()
    seed = 2025

    hrex_params = HREXParams(n_frames_bisection=10, optimize_target_overlap=0.999)
    md_params = MDParams(n_frames=10, n_eq_steps=100, steps_per_frame=100, seed=seed, hrex_params=hrex_params)
    with pytest.warns(UserWarning, match="unreliable starting point"):
        estimate_relative_free_energy_bisection_hrex(
            mol_a,
            mol_b,
            core,
            forcefield,
            None,
            md_params=md_params,
            prefix="low_overlap",
            n_windows=3,
        )


@patch("timemachine.fe.rbfe.compute_u_kn")
def test_rebalance_lambda_schedule(mock_compute_u_kn):
    # has some issues with default PyMBAR settings
    with path_to_internal_file("timemachine.testsystems.data", "u_kn_unstable.npz") as path_to_npz:
        npz = np.load(path_to_npz)
        u_kn, N_k, initial_lambdas = npz["u_kn"], npz["N_k"], npz["initial_lambdas"]

    mock_compute_u_kn.return_value = (u_kn, N_k)

    final_lambdas = []

    def initial_state_fxn(lamb: float):
        final_lambdas.append(lamb)
        return

    initial_states = [Mock(lamb=lamb) for lamb in initial_lambdas]
    trajectories = [None for _ in range(len(initial_states))]
    rebalance_lambda_schedule(initial_states, initial_state_fxn, trajectories, 2 / 3)
    assert len(final_lambdas) > 35
    assert len(final_lambdas) < len(initial_lambdas)
    assert np.max(np.diff(final_lambdas)) < 0.05


if __name__ == "__main__":
    # convenience: so we can run this directly from python tests/test_relative_free_energy.py without
    # toggling the pytest marker
    test_run_hif2a_test_system(estimate_relative_free_energy)
    test_run_hif2a_test_system(estimate_relative_free_energy_bisection)
    test_run_hif2a_test_system(estimate_relative_free_energy_bisection_hrex)
