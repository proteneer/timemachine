import numpy as np

from typing import Tuple, Dict, List

from md.states import CoordsVelBox
from md.moves import CompoundMove

from md.thermostat.moves import UnadjustedLangevinMove
from md.barostat.moves import MonteCarloBarostat

from md.utils import compute_box_volume


def simulate_npt_traj(
    thermostat: UnadjustedLangevinMove, barostat: MonteCarloBarostat, initial_state: CoordsVelBox, n_moves=1000
) -> Tuple[List[CoordsVelBox], Dict]:
    barostat.reset()

    # alternate between thermostat moves and barostat moves
    traj = [initial_state]
    volume_traj = [compute_box_volume(traj[0].box)]
    proposal_scale_traj = [barostat.max_delta_volume]

    compound_move = CompoundMove([thermostat, barostat])

    for _ in range(n_moves):
        traj.append(compound_move.move(traj[-1]))

        # accumulate result trajectories
        volume_traj.append(compute_box_volume(traj[-1].box))
        proposal_scale_traj.append(barostat.max_delta_volume)

    extras = dict(
        volume_traj=np.array(volume_traj),
        proposal_scale_traj=np.array(proposal_scale_traj),
    )

    return traj, extras
