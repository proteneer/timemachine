# test that we can run relative free energy simulations in complex and in solvent
# this doesn't test for accuracy, just that everything mechanically runs.
import numpy as np
import pytest

from timemachine.fe.rbfe import HostConfig, estimate_relative_free_energy
from timemachine.md import builders
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


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

    assert solvent_res.plot_png is not None
    assert abs(np.sum(solvent_res.all_dGs)) < 10.0
    assert np.linalg.norm(solvent_res.all_dGs) < 10.0
    assert len(solvent_res.frames[0] == n_frames)
    assert len(solvent_res.frames[-1] == n_frames)
    assert len(solvent_res.boxes[0] == n_frames)
    assert len(solvent_res.boxes[-1] == n_frames)
    assert solvent_res.lambda_schedule == lambda_schedule

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

    assert complex_res.plot_png is not None
    assert abs(np.sum(complex_res.all_dGs)) < 10.0
    assert np.linalg.norm(complex_res.all_dGs) < 10.0
    assert len(complex_res.frames[0]) == n_frames
    assert len(complex_res.frames[-1]) == n_frames
    assert len(complex_res.boxes[0]) == n_frames
    assert len(complex_res.boxes[-1]) == n_frames
    assert complex_res.lambda_schedule == lambda_schedule


@pytest.mark.nightly(reason="Slow!")
def test_run_hif2a_test_system():

    st = get_hif2a_ligand_pair_single_topology()
    mol_a = st.mol_a
    mol_b = st.mol_b
    core = st.core
    forcefield = st.ff
    protein_path = "tests/data/hif2a_nowater_min.pdb"

    run_pair(mol_a, mol_b, core, forcefield, n_frames=100, protein_path=protein_path)
