# test that we can run relative free energy simulations in complex and in solvent
# this doesn't test for accuracy, just that everything mechanically runs.
from importlib import resources

import numpy as np
import pytest

from timemachine.fe.rbfe import HostConfig, SimulationResult, estimate_relative_free_energy, sample
from timemachine.md import builders
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


def run_bitwise_reproducibility(mol_a, mol_b, core, forcefield, n_frames):
    # test that we can bitwise reproduce our trajectory using the initial state information

    lambda_schedule = [0.01, 0.02, 0.03]
    seed = 2023
    box_width = 4.0
    solvent_sys, solvent_conf, solvent_box, _ = builders.build_water_system(box_width)
    solvent_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    solvent_host_config = HostConfig(solvent_sys, solvent_conf, solvent_box)
    keep_idxs = [0, 1, 2]
    solvent_res = estimate_relative_free_energy(
        mol_a,
        mol_b,
        core,
        forcefield,
        solvent_host_config,
        seed,
        n_frames=n_frames,
        prefix="solvent",
        lambda_schedule=lambda_schedule,
        keep_idxs=keep_idxs,
    )

    all_frames, all_boxes = [], []
    for state in solvent_res.initial_states:
        frames, boxes = sample(state, solvent_res.protocol)
        all_frames.append(frames)
        all_boxes.append(boxes)

    np.testing.assert_equal(solvent_res.frames, all_frames)
    np.testing.assert_equal(solvent_res.boxes, all_boxes)


def run_pair(mol_a, mol_b, core, forcefield, n_frames, protein_path):

    lambda_schedule = [0.01, 0.02, 0.03]

    seed = 2023
    box_width = 4.0
    solvent_sys, solvent_conf, solvent_box, _ = builders.build_water_system(box_width)
    solvent_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    solvent_host_config = HostConfig(solvent_sys, solvent_conf, solvent_box)
    solvent_res = estimate_relative_free_energy(
        mol_a,
        mol_b,
        core,
        forcefield,
        solvent_host_config,
        seed,
        n_frames=n_frames,
        prefix="solvent",
        lambda_schedule=lambda_schedule,
    )

    assert solvent_res.overlap_summary_png is not None
    assert solvent_res.overlap_detail_png is not None
    assert abs(np.sum(solvent_res.all_dGs)) < 10.0
    assert np.linalg.norm(solvent_res.all_dGs) < 10.0
    assert len(solvent_res.frames[0] == n_frames)
    assert len(solvent_res.frames[-1] == n_frames)
    assert len(solvent_res.boxes[0] == n_frames)
    assert len(solvent_res.boxes[-1] == n_frames)
    assert [x.lamb for x in solvent_res.initial_states] == lambda_schedule
    assert solvent_res.protocol.n_frames == n_frames

    def check_overlaps(result: SimulationResult):
        assert result.overlaps_by_lambda.shape == (len(lambda_schedule) - 1,)
        assert result.overlaps_by_lambda_by_component.shape[1] == len(lambda_schedule) - 1
        for overlaps in [result.overlaps_by_lambda, result.overlaps_by_lambda_by_component]:
            assert (0.0 < overlaps).all()
            assert (overlaps < 0.5).all()

    check_overlaps(solvent_res)

    seed = 2024
    complex_sys, complex_conf, _, _, complex_box, _ = builders.build_protein_system(protein_path)
    complex_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes
    complex_host_config = HostConfig(complex_sys, complex_conf, complex_box)
    complex_res = estimate_relative_free_energy(
        mol_a,
        mol_b,
        core,
        forcefield,
        complex_host_config,
        seed,
        n_frames=n_frames,
        prefix="complex",
        lambda_schedule=lambda_schedule,
    )

    assert solvent_res.overlap_summary_png is not None
    assert complex_res.overlap_detail_png is not None
    assert abs(np.sum(complex_res.all_dGs)) < 10.0
    assert np.linalg.norm(complex_res.all_dGs) < 10.0
    assert len(complex_res.frames[0]) == n_frames
    assert len(complex_res.frames[-1]) == n_frames
    assert len(complex_res.boxes[0]) == n_frames
    assert len(complex_res.boxes[-1]) == n_frames
    assert [x.lamb for x in complex_res.initial_states] == lambda_schedule
    assert complex_res.protocol.n_frames == n_frames

    check_overlaps(complex_res)


@pytest.mark.nightly(reason="Slow!")
def test_run_hif2a_test_system():

    st = get_hif2a_ligand_pair_single_topology()
    mol_a = st.mol_a
    mol_b = st.mol_b
    core = st.core
    forcefield = st.ff

    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as protein_path:
        run_pair(mol_a, mol_b, core, forcefield, n_frames=100, protein_path=str(protein_path))
    run_bitwise_reproducibility(mol_a, mol_b, core, forcefield, n_frames=100)


if __name__ == "__main__":
    # convenience: so we can run this directly from python tests/test_relative_free_energy.py without
    # toggling the pytest marker
    test_run_hif2a_test_system()
