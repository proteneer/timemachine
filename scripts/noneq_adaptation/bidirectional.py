from md.thermostat.utils import sample_velocities
from md.thermostat.moves import UnadjustedLangevinMove
from md.states import CoordsVelBox
import os
from simtk import unit
import numpy as np
from scipy.optimize import root_scalar

from tqdm import tqdm
from typing import List
from pymbar import EXP

from md.states import CoordsVelBox
from fe.free_energy import construct_lambda_schedule

from simtk import unit
import numpy as np
from scipy.optimize import root_scalar

from timemachine.lib import LangevinIntegrator, custom_ops
from tqdm import tqdm

from typing import List
from functools import partial

from pymbar import EXP

import os

# from same script folder...
from testsystem import (
    temperature, coords, masses, complex_box,
    integrator_impl, ensemble, potential_energy_model,
)
from adapt_noneq import optimized_lam_traj_path, sample_at_equilibrium

from deploy import interpolate_lambda_schedule, noneq_du_dl


def noneq_move(x: CoordsVelBox, lambda_schedule: np.array) -> CoordsVelBox:
    """Run a nonequilibrium trajectory, storing final state"""
    ctxt = custom_ops.Context(x.coords, x.velocities, x.box, integrator_impl, potential_energy_model.all_impls)

    # arguments: lambda_schedule, du_dl_interval, x_interval
    _, _ = ctxt.multiple_steps(lambda_schedule, 0, 0)

    return CoordsVelBox(ctxt.get_x_t(), ctxt.get_v_t(), x.box.copy())



if __name__ == '__main__':

    lam_traj = np.load(optimized_lam_traj_path)

    # generate end-state samples
    n_equil_steps = 10000
    n_samples = 100

    v_0 = sample_velocities(masses * unit.amu, temperature)
    initial_state = CoordsVelBox(coords, v_0, complex_box)

    print('equilibrating...')
    thermostat_0 = UnadjustedLangevinMove(
        integrator_impl, potential_energy_model.all_impls,
        lam=0.0, n_steps=n_equil_steps
    )
    equilibrated_0 = thermostat_0.move(initial_state)

    print(f'collecting {n_samples} samples from lam=0...')
    samples_0 = sample_at_equilibrium(equilibrated_0, lam=0.0, n_samples=n_samples)

    print('switching from lam=0 -> lam=1 to initialize lam=1 equilibrium sampling...')
    approx_equilibrated_1 = noneq_move(samples_0[-1], interpolate_lambda_schedule(lam_traj, n_equil_steps))
    thermostat_1 = UnadjustedLangevinMove(
        integrator_impl, potential_energy_model.all_impls,
        lam=1.0, n_steps=n_equil_steps
    )
    equilibrated_1 = thermostat_1.move(approx_equilibrated_1)

    print(f'collecting {n_samples} samples from lam=1...')
    samples_1 = sample_at_equilibrium(equilibrated_1, lam=1.0)
