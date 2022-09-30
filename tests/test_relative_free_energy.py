# test that we can run relative free energy simulations in complex and in solvent
# this doesn't test for accuracy, just that everything mechanically runs.
from importlib import resources

import numpy as np
import pytest

from timemachine.constants import DEFAULT_FF
from timemachine.fe.rbfe import (
    HostConfig,
    SimulationResult,
    estimate_relative_free_energy,
    pair_overlap_from_ukln,
    sample,
)
from timemachine.ff import Forcefield
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


def run_triple(mol_a, mol_b, core, forcefield, n_frames, protein_path, n_eq_steps):

    seed = 2023

    lambda_schedule = [0.01, 0.02, 0.03]
    vacuum_host_config = None
    vacuum_res = estimate_relative_free_energy(
        mol_a,
        mol_b,
        core,
        forcefield,
        vacuum_host_config,
        seed,
        n_frames=n_frames,
        prefix="vacuum",
        lambda_schedule=lambda_schedule,
        n_eq_steps=n_eq_steps,
    )

    assert vacuum_res.overlap_summary_png is not None
    assert vacuum_res.overlap_detail_png is not None
    assert np.linalg.norm(vacuum_res.all_errs) < 0.1
    assert len(vacuum_res.frames[0] == n_frames)
    assert len(vacuum_res.frames[-1] == n_frames)
    assert len(vacuum_res.boxes[0] == n_frames)
    assert len(vacuum_res.boxes[-1] == n_frames)
    assert [x.lamb for x in vacuum_res.initial_states] == lambda_schedule
    assert vacuum_res.protocol.n_frames == n_frames
    assert vacuum_res.protocol.n_eq_steps == n_eq_steps

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
        n_eq_steps=n_eq_steps,
    )

    assert solvent_res.overlap_summary_png is not None
    assert solvent_res.overlap_detail_png is not None
    assert np.linalg.norm(solvent_res.all_errs) < 0.1
    assert len(solvent_res.frames[0] == n_frames)
    assert len(solvent_res.frames[-1] == n_frames)
    assert len(solvent_res.boxes[0] == n_frames)
    assert len(solvent_res.boxes[-1] == n_frames)
    assert [x.lamb for x in solvent_res.initial_states] == lambda_schedule
    assert solvent_res.protocol.n_frames == n_frames
    assert solvent_res.protocol.n_eq_steps == n_eq_steps

    def check_overlaps(result: SimulationResult):
        assert result.overlaps_by_lambda.shape == (len(lambda_schedule) - 1,)
        assert result.overlaps_by_lambda_by_component.shape[1] == len(lambda_schedule) - 1
        for overlaps in [result.overlaps_by_lambda, result.overlaps_by_lambda_by_component]:
            assert (0.0 < overlaps).all()
            assert (overlaps < 1.0).all()

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
        n_eq_steps=n_eq_steps,
    )

    assert solvent_res.overlap_summary_png is not None
    assert complex_res.overlap_detail_png is not None
    assert np.linalg.norm(complex_res.all_errs) < 0.1
    assert len(complex_res.frames[0]) == n_frames
    assert len(complex_res.frames[-1]) == n_frames
    assert len(complex_res.boxes[0]) == n_frames
    assert len(complex_res.boxes[-1]) == n_frames
    assert [x.lamb for x in complex_res.initial_states] == lambda_schedule
    assert complex_res.protocol.n_frames == n_frames
    assert complex_res.protocol.n_eq_steps == n_eq_steps

    check_overlaps(complex_res)


@pytest.mark.nightly(reason="Slow!")
def test_run_hif2a_test_system():

    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file(DEFAULT_FF)

    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as protein_path:
        run_triple(mol_a, mol_b, core, forcefield, n_frames=100, protein_path=str(protein_path), n_eq_steps=1000)
    run_bitwise_reproducibility(mol_a, mol_b, core, forcefield, n_frames=100)


@pytest.mark.nogpu
def test_pair_overlap_from_ukln():
    def gaussian_overlap(p1, p2):
        def make_gaussian(params):
            mu, sigma = params

            def u(x):
                return (x - mu) ** 2 / (2 * sigma ** 2)

            rng = np.random.default_rng(2022)
            x = rng.normal(mu, sigma, 100)

            return u, x

        u1, x1 = make_gaussian(p1)
        u2, x2 = make_gaussian(p2)

        u_kln = np.array([[u1(x1), u1(x2)], [u2(x1), u2(x2)]])

        return pair_overlap_from_ukln(u_kln)

    # identical distributions
    np.testing.assert_allclose(gaussian_overlap((0, 1), (0, 1)), 1.0)

    # non-overlapping
    assert gaussian_overlap((0, 0.01), (1, 0.01)) < 1e-10

    # overlapping
    assert gaussian_overlap((0, 0.1), (0.5, 0.2)) > 0.1


if __name__ == "__main__":
    # convenience: so we can run this directly from python tests/test_relative_free_energy.py without
    # toggling the pytest marker
    test_run_hif2a_test_system()
