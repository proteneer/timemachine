import numpy as np

from timemachine import testsystems
from timemachine.fe import absolute_hydration
from timemachine.fe.free_energy import MDParams
from timemachine.ff import Forcefield


def test_run_solvent_absolute_hydration():
    seed = 2022
    n_frames = 10
    n_eq_steps = 100
    n_windows = 8
    steps_per_frame = 10
    mol, _ = testsystems.ligands.get_biphenyl()
    ff = Forcefield.load_default()
    md_params = MDParams(seed=seed, n_eq_steps=n_eq_steps, n_frames=n_frames, steps_per_frame=steps_per_frame)
    res, host_config = absolute_hydration.run_solvent(mol, ff, None, md_params=md_params, n_windows=n_windows)

    assert res.plots.overlap_summary_png is not None
    assert res.plots.overlap_detail_png is not None
    assert np.linalg.norm(res.final_result.dG_errs) < 20.0
    assert len(res.frames) == n_windows
    assert len(res.boxes) == n_windows
    assert len(res.frames[0]) == n_frames
    assert len(res.frames[-1]) == n_frames
    assert len(res.boxes[0]) == n_frames
    assert len(res.boxes[-1]) == n_frames
    assert res.md_params == md_params
    assert host_config.omm_system is not None
    # The number of waters in the system should stay constant
    assert host_config.num_water_atoms == 6282
    assert host_config.conf.shape == (res.frames[0][0].shape[0] - mol.GetNumAtoms(), 3)
    assert host_config.box.shape == (3, 3)
