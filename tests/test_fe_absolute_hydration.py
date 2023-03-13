import numpy as np

from timemachine import testsystems
from timemachine.constants import DEFAULT_FF
from timemachine.fe import absolute_hydration
from timemachine.ff import Forcefield


def test_run_solvent():
    seed = 2022
    n_frames = 10
    n_eq_steps = 100
    n_windows = 8
    mol, _ = testsystems.ligands.get_biphenyl()
    ff = Forcefield.load_from_file(DEFAULT_FF)
    res, top, host_config = absolute_hydration.run_solvent(
        mol, ff, None, n_frames, seed, n_eq_steps=n_eq_steps, n_windows=n_windows
    )

    assert res.plots.overlap_summary_png is not None
    assert res.plots.overlap_detail_png is not None
    assert np.linalg.norm([r.dG_err for r in res.final_result.bar_results]) < 10
    assert len(res.frames) == 2
    assert len(res.boxes) == 2
    assert len(res.frames[0]) == n_frames
    assert len(res.frames[-1]) == n_frames
    assert len(res.boxes[0]) == n_frames
    assert len(res.boxes[-1]) == n_frames
    assert res.md_params.n_frames == n_frames
    assert res.md_params.n_eq_steps == n_eq_steps
    assert host_config.omm_system is not None
    assert host_config.conf.shape == (6282, 3)
    assert host_config.box.shape == (3, 3)
