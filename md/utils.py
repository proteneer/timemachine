import numpy as np
from tqdm import tqdm

from typing import Tuple, Dict, List

from md.states import CoordsVelBox
from md.ensembles import NPTEnsemble

from md.thermostat.moves import UnadjustedMDMove
from md.moves import CompoundMove
from md.barostat.moves import MonteCarloBarostat
from md.barostat.utils import compute_box_volume

from timemachine.lib import custom_ops


def run_thermostatted_md(
        integrator_impl, bound_impls, initial_state: CoordsVelBox,
        lam: float, n_steps=5) -> Tuple[np.array, np.array]:

    # note: context creation overhead here is actually very small!
    ctxt = custom_ops.Context(
        initial_state.coords,
        initial_state.velocities,
        initial_state.box,
        integrator_impl,
        bound_impls,
    )

    # arguments: lambda_schedule, du_dl_interval, x_interval
    _, _ = ctxt.multiple_steps(lam * np.ones(n_steps), 0, 0)
    x_t = ctxt.get_x_t()
    v_t = ctxt.get_v_t()

    return x_t, v_t


def simulate_npt_traj(
        ensemble: NPTEnsemble, thermostat, barostat: MonteCarloBarostat,
        initial_state: CoordsVelBox, n_moves=1000) -> Tuple[List[CoordsVelBox], Dict]:

    barostat.reset()

    # alternate between thermostat moves and barostat moves
    traj = [initial_state]
    volume_traj = [compute_box_volume(traj[0].box)]
    proposal_scale_traj = [barostat.max_delta_volume]

    trange = tqdm(range(n_moves))

    bound_impls = ensemble.potential_energy.all_impls

    compound_move = CompoundMove([thermostat, barostat])

    for _ in trange:
        traj.append(compound_move.move(traj[-1]))

        # accumulate result trajectories
        volume_traj.append(compute_box_volume(traj[-1].box))
        proposal_scale_traj.append(barostat.max_delta_volume)

        # informative progress bar
        trange.set_postfix(
            volume=f'{volume_traj[-1]:.3f}',
            acceptance_fraction=f'{barostat.acceptance_fraction:.3f}',
            proposal_scale=f'{barostat.max_delta_volume:.3f}',
        )

    extras = dict(
        volume_traj=np.array(volume_traj),
        proposal_scale_traj=np.array(proposal_scale_traj),
    )

    return traj, extras
