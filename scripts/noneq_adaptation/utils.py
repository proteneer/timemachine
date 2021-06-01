from pymbar import EXP
import numpy as np
from typing import Callable, List
from md.states import CoordsVelBox

Lambda = float
Energies = np.array
VectorizedPotentialEnergy = Callable[[List[CoordsVelBox], Lambda], Energies]
Samples = List[CoordsVelBox]


def compute_work_increments(
        u_vec: VectorizedPotentialEnergy,
        sample_traj: List[Samples],
        lam_traj: List[Lambda]) -> np.array:
    """For computing work via sum of u(x, lam[t+1]) - u(x, lam[t]) increments"""

    work_increments = []
    for (X, lam_init, lam_final) in zip(sample_traj, lam_traj[:-1], lam_traj[1:]):
        work_increments.append(u_vec(X, lam_final) - u_vec(X, lam_init))
    work_increments = np.array(work_increments)
    works = np.sum(work_increments, 0)
    print(f'stddev(w_f): {np.std(works):.3f} kBT')
    print(f'EXP(w_f): {EXP(works)[0]:.3f} kBT')
    print('(with work computed via w = sum_t u(x_t, lam[t+1]) - u(x_t, lam[t])')

    return work_increments
