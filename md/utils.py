import numpy as np
from tqdm import tqdm
from time import time

from typing import Tuple, Dict

from md.states import CoordsVelBox
from md.ensembles import NPTEnsemble
from md.barostat.moves import MonteCarloBarostat
from md.barostat.utils import compute_box_volume

from timemachine.lib import custom_ops


def run_thermostatted_md(
        integrator_impl, bound_impls, initial_state: CoordsVelBox,
        lam: float, n_steps=5) -> Tuple[np.array, np.array]:


    ctxt = custom_ops.Context(
        initial_state.coords,
        initial_state.velocities,
        initial_state.box,
        integrator_impl,
        bound_impls
    )

    # arguments: lambda_schedule, du_dl_interval, x_interval
    _, _ = ctxt.multiple_steps(lam * np.ones(n_steps), 0, 0)
    x_t = ctxt.get_x_t()
    v_t = ctxt.get_v_t()

    return x_t, v_t


def simulate_npt_traj(
        ensemble: NPTEnsemble, integrator_impl, barostat: MonteCarloBarostat,
        initial_state: CoordsVelBox,
        lam=1.0, n_moves=1000, barostat_interval=5) -> Tuple[np.array, np.array, Dict]:
    """TODO: replace with more modular design: composition of [MDMove, MCBarostatMove]"""
    barostat.reset()

    # alternate between thermostat moves and barostat moves
    traj = [initial_state]
    volume_traj = [compute_box_volume(traj[0].box)]
    proposal_scale_traj = [barostat.max_delta_volume]

    trange = tqdm(range(n_moves))

    bound_impls = ensemble.potential_energy.all_impls

    v_t = initial_state.velocities.copy()
    for _ in trange:
        t0 = time()

        # MDMove
        x_t, v_t = run_thermostatted_md(
            integrator_impl, bound_impls, traj[-1], lam, n_steps=barostat_interval)
        after_nvt = CoordsVelBox(x_t, v_t, traj[-1].box.copy())

        t1 = time()

        # MCBarostatMove
        after_npt = barostat.move(after_nvt)

        t2 = time()

        # accumulate result trajectories
        traj.append(after_npt)
        volume_traj.append(compute_box_volume(after_npt.box))
        proposal_scale_traj.append(barostat.max_delta_volume)

        # informative progress bar
        trange.set_postfix(volume=f'{volume_traj[-1]:.3f}',
                           acceptance_fraction=f'{barostat.acceptance_fraction:.3f}',
                           md_proposal_time=f'{(t1 - t0):.3f}s',
                           barostat_proposal_time=f'{(t2 - t1):.3f}s',
                           proposal_scale=f'{barostat.max_delta_volume:.3f}',
                           )

    # TODO: make this an MDTraj trajectory?
    x_traj = np.array([snapshot.coords for snapshot in traj])
    box_traj = np.array([snapshot.box for snapshot in traj])

    volume_traj = np.array(volume_traj)
    proposal_scale_traj = np.array(proposal_scale_traj)

    extras = dict(volume_traj=volume_traj, proposal_scale_traj=proposal_scale_traj)

    return x_traj, box_traj, extras
