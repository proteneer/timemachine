# test that we can run relative free energy simulations in complex and in solvent
# this doesn't test for accuracy, just that everything mechanically runs.
from importlib import resources
from typing import List

import numpy as np
import pytest

from timemachine.constants import DEFAULT_FF
from timemachine.fe.free_energy import HostConfig, PairBarResult, SimulationResult, image_frames, sample
from timemachine.fe.rbfe import (
    estimate_relative_free_energy,
    estimate_relative_free_energy_via_greedy_bisection,
    run_solvent,
    run_vacuum,
)
from timemachine.ff import Forcefield
from timemachine.md import builders
from timemachine.md.barostat.utils import compute_box_center
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


def run_bitwise_reproducibility(mol_a, mol_b, core, forcefield, n_frames, estimate_relative_free_energy_fn):
    # test that we can bitwise reproduce our trajectory using the initial state information

    seed = 2023
    box_width = 4.0
    n_windows = 4
    solvent_sys, solvent_conf, solvent_box, _ = builders.build_water_system(box_width, forcefield.water_ff)
    solvent_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    solvent_host_config = HostConfig(solvent_sys, solvent_conf, solvent_box)
    keep_idxs = list(range(n_windows))

    solvent_res = estimate_relative_free_energy_fn(
        mol_a,
        mol_b,
        core,
        forcefield,
        solvent_host_config,
        seed,
        n_frames=n_frames,
        prefix="solvent",
        lambda_interval=(0.01, 0.03),
        n_windows=n_windows,
        keep_idxs=keep_idxs,
    )

    all_frames, all_boxes = [], []
    for state in solvent_res.final_result.initial_states:
        frames, boxes = sample(state, solvent_res.md_params)
        all_frames.append(frames)
        all_boxes.append(boxes)

    np.testing.assert_equal(solvent_res.frames, all_frames)
    np.testing.assert_equal(solvent_res.boxes, all_boxes)


def run_triple(mol_a, mol_b, core, forcefield, n_frames, protein_path, n_eq_steps, estimate_relative_free_energy_fn):

    seed = 2023
    lambda_interval = [0.01, 0.03]
    n_windows = 3

    def check_sim_result(sim_res: SimulationResult, state_seeds: List[int]):
        assert len(sim_res.final_result.initial_states) == n_windows
        assert sim_res.final_result.initial_states[0].lamb == lambda_interval[0]
        assert sim_res.final_result.initial_states[-1].lamb == lambda_interval[1]
        assert [initial_state.integrator.seed for initial_state in sim_res.final_result.initial_states] == state_seeds
        assert sim_res.plots.dG_errs_png is not None
        assert sim_res.plots.overlap_summary_png is not None
        assert sim_res.plots.overlap_detail_png is not None

        assert len(sim_res.frames[0]) == n_frames
        assert len(sim_res.frames[-1]) == n_frames
        assert len(sim_res.boxes[0]) == n_frames
        assert len(sim_res.boxes[-1]) == n_frames
        assert sim_res.md_params.n_frames == n_frames
        assert sim_res.md_params.n_eq_steps == n_eq_steps

        def check_pair_bar_result(res: PairBarResult):
            n_pairs = len(res.initial_states) - 1
            assert len(res.bar_results) == n_pairs

            for dg_errs in [res.dG_errs, res.dG_err_by_component_by_lambda]:
                assert np.all(0.0 < np.asarray(dg_errs))
                assert np.linalg.norm(dg_errs) < 0.1

            assert res.overlap_by_component_by_lambda.shape[0] == n_pairs
            assert res.overlap_by_component_by_lambda.shape[1] == res.dG_err_by_component_by_lambda.shape[1]
            for overlaps in [res.overlaps, res.overlap_by_component_by_lambda]:
                assert np.all(0.0 < np.asarray(overlaps))
                assert np.all(np.asarray(overlaps) < 1.0)

        check_pair_bar_result(sim_res.final_result)
        for res in sim_res.intermediate_results:
            check_pair_bar_result(res)

    vacuum_res = estimate_relative_free_energy_fn(
        mol_a,
        mol_b,
        core,
        forcefield,
        host_config=None,
        seed=seed,
        n_frames=n_frames,
        prefix="vacuum",
        lambda_interval=lambda_interval,
        n_windows=n_windows,
        n_eq_steps=n_eq_steps,
    )
    print("vacuum")
    is_bisection = estimate_relative_free_energy_fn == estimate_relative_free_energy_via_greedy_bisection
    state_seeds = [6998, 6082, 36] if is_bisection else [6998, 3540, 36]
    check_sim_result(vacuum_res, state_seeds=state_seeds)

    box_width = 4.0
    solvent_sys, solvent_conf, solvent_box, _ = builders.build_water_system(box_width, forcefield.water_ff)
    solvent_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    solvent_host_config = HostConfig(solvent_sys, solvent_conf, solvent_box)
    solvent_res = estimate_relative_free_energy_fn(
        mol_a,
        mol_b,
        core,
        forcefield,
        solvent_host_config,
        seed,
        n_frames=n_frames,
        prefix="solvent",
        lambda_interval=lambda_interval,
        n_windows=n_windows,
        n_eq_steps=n_eq_steps,
    )

    print("solvent")
    state_seeds = [8783, 8019, 3494] if is_bisection else [8783, 701, 3494]
    check_sim_result(solvent_res, state_seeds=state_seeds)

    seed = 2024
    complex_sys, complex_conf, _, _, complex_box, _ = builders.build_protein_system(
        protein_path, forcefield.protein_ff, forcefield.water_ff
    )
    complex_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    complex_host_config = HostConfig(complex_sys, complex_conf, complex_box)
    complex_res = estimate_relative_free_energy_fn(
        mol_a,
        mol_b,
        core,
        forcefield,
        complex_host_config,
        seed,
        n_frames=n_frames,
        prefix="complex",
        lambda_interval=lambda_interval,
        n_windows=n_windows,
        n_eq_steps=n_eq_steps,
    )

    print("complex")
    state_seeds = [9977, 3195, 5508] if is_bisection else [9977, 1713, 5508]
    check_sim_result(complex_res, state_seeds=state_seeds)


# @pytest.mark.nightly(reason="Slow!")
@pytest.mark.parametrize(
    "estimate_relative_free_energy_fn",
    [estimate_relative_free_energy, estimate_relative_free_energy_via_greedy_bisection],
)
def test_run_hif2a_test_system(estimate_relative_free_energy_fn):

    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file(DEFAULT_FF)

    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as protein_path:
        run_triple(
            mol_a,
            mol_b,
            core,
            forcefield,
            n_frames=100,
            protein_path=str(protein_path),
            n_eq_steps=1000,
            estimate_relative_free_energy_fn=estimate_relative_free_energy_fn,
        )
    run_bitwise_reproducibility(
        mol_a,
        mol_b,
        core,
        forcefield,
        n_frames=100,
        estimate_relative_free_energy_fn=estimate_relative_free_energy_fn,
    )


def test_steps_per_frames():
    """Verifies that modifying steps per frames doesn't change result trajectory"""
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file(DEFAULT_FF)
    seed = 2022
    frames = 5
    res = run_vacuum(mol_a, mol_b, core, forcefield, None, frames, seed, n_eq_steps=10, steps_per_frame=2, n_windows=2)
    assert len(res.frames[0]) == frames

    frames = 2
    test_res = run_vacuum(
        mol_a, mol_b, core, forcefield, None, frames, seed, n_eq_steps=10, steps_per_frame=5, n_windows=2
    )
    assert len(test_res.frames[0]) == frames
    assert len(test_res.frames) == 2
    # The last frame from the trajectories should match as num_frames * steps_per_frame are equal
    for frame, test_frame in zip(res.frames, test_res.frames):
        np.testing.assert_array_equal(frame[-1], test_frame[-1])


def test_imaging_frames():
    """Verify that imaging frames places ligand at center and all coordinates are close to being within the box.

    Does not check precision, as it is known to be lossy. Only to be used for post-processing/visualization."""
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file(DEFAULT_FF)
    seed = 2022
    frames = 1
    steps_per_frame = 1
    equil_steps = 1
    windows = 2
    res, _, _ = run_solvent(
        mol_a,
        mol_b,
        core,
        forcefield,
        None,
        frames,
        seed,
        n_eq_steps=equil_steps,
        steps_per_frame=steps_per_frame,
        n_windows=windows,
    )
    keep_idxs = [0, len(res.final_result.initial_states) - 1]
    assert len(keep_idxs) == len(res.frames)

    # A buffer, as imaging doesn't ensure everything is perfectly in the box
    padding = 0.3
    for i, (frames, boxes) in enumerate(zip(res.frames, res.boxes)):
        initial_state = res.final_result.initial_states[keep_idxs[i]]
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
    [estimate_relative_free_energy, estimate_relative_free_energy_via_greedy_bisection],
)
def test_rbfe_with_1_window(estimate_relative_free_energy_fn):
    """Should not be able to run a relative free energy calculation with a single window"""
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file(DEFAULT_FF)
    seed = 2022
    with pytest.raises(AssertionError):
        estimate_relative_free_energy_fn(
            mol_a,
            mol_b,
            core,
            forcefield,
            None,
            seed,
            n_frames=1,
            prefix="failure",
            n_windows=1,
            steps_per_frame=1,
            n_eq_steps=10,
        )


if __name__ == "__main__":
    # convenience: so we can run this directly from python tests/test_relative_free_energy.py without
    # toggling the pytest marker
    test_run_hif2a_test_system(estimate_relative_free_energy)
    test_run_hif2a_test_system(estimate_relative_free_energy_via_greedy_bisection)
