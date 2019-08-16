import numpy as np
import random

from system import forcefield
from timemachine.lib import custom_ops
from timemachine.integrator import langevin_coefficients

from timemachine import constants

from simtk.openmm.app import PDBFile, DCDFile, Topology

import scipy

def average_E_and_derivatives(reservoir):
    """
    Compute the average energy and derivatives

    Parameters
    ----------
    reservoir: list of reservoir
        [
            [E, dE_dx, dx_dp, dE_dp, x],
            [E, dE_dx, dx_dp, dE_dp, x],
            ...
        ]

    Returns
    -------
    Average energy, analytic total derivative, and thermodynamic gradient

    """
    if len(reservoir) == 0:
        return np.nan, 0, 0

    running_sum_total_derivs = None
    running_sum_E = 0
    n_reservoir = len(reservoir)

    running_sum_dE_dp = None
    running_sum_EmultdE_dp = None

    for E, dE_dx, dx_dp, dE_dp, _ in reservoir:
        if running_sum_total_derivs is None:
            running_sum_total_derivs = np.zeros_like(dE_dp)
        if running_sum_dE_dp is None:
            running_sum_dE_dp = np.zeros_like(dE_dp)
        if running_sum_EmultdE_dp is None:
            running_sum_EmultdE_dp = np.zeros_like(dE_dp)

        if np.isnan(E):
            n_reservoir -= 1
        else:
            # tensor contract [N,3] with [P, N, 3] and add dE_dp for a shape P array
            total_dE_dp = np.einsum('kl,mkl->m', dE_dx, dx_dp) + dE_dp
            running_sum_total_derivs += total_dE_dp
            running_sum_E += E

            running_sum_dE_dp += dE_dp
            running_sum_EmultdE_dp += E*dE_dp

        if n_reservoir < 1:
            return np.nan, 0, 0

    # compute the thermodynamic average: boltz*(<E><dE/dp> - <E.dE/dp>)
    thermo_deriv = running_sum_E*running_sum_dE_dp - running_sum_EmultdE_dp

    return running_sum_E/n_reservoir, running_sum_total_derivs/n_reservoir, -constants.BOLTZ*(thermo_deriv/n_reservoir)/(100)


def run_simulation(
    potentials,
    params,
    param_groups,
    conf,
    masses,
    dp_idxs,
    n_samples=200,
    start_dt=1e-6,
    end_dt=1e-2,
    scale=1.05,
    convergence_tolerance=10
    ):

    num_atoms = len(masses)

    potentials = forcefield.merge_potentials(potentials)

    dt = start_dt
    ca, cb, cc = langevin_coefficients(
        temperature=25.0,
        dt=dt,
        friction=100,
        masses=masses
    )

    # m_dt, m_ca, m_cb, m_cc = dt, 0.5, cb, np.zeros_like(masses)

    m_dt, m_ca, m_cb, m_cc = dt, 0.9, np.ones_like(cb)/10000, np.zeros_like(masses)

    opt = custom_ops.LangevinOptimizer_f64(
        m_dt,
        m_ca,
        m_cb.astype(np.float64),
        m_cc.astype(np.float64)
    )

    v0 = np.zeros_like(conf)
    dp_idxs = dp_idxs.astype(np.int32)

    ctxt = custom_ops.Context_f64(
        potentials,
        opt,
        params.astype(np.float64),
        conf.astype(np.float64), # x0
        v0.astype(np.float64), # v0
        dp_idxs
    )

    tolerance = convergence_tolerance

    def mean_norm(conf):
        norm_x = np.dot(conf.reshape(-1), conf.reshape(-1))/num_atoms
        return np.sqrt(norm_x)

    # set normalized convergence criteria
    x_norm = mean_norm(conf)
    x_norm = np.where(x_norm < 1, 1, x_norm)
    epsilon = tolerance/x_norm

    max_iter = 10000
    for i in range(max_iter):
        # adjust dt by a scale factor for each time step up until it reaches a maximum value
        dt *= scale
        dt = min(dt, end_dt)
        opt.set_dt(dt)
        ctxt.step()
        if i > 50 and i % 100 == 0:
            if np.isnan(ctxt.get_E()):
                final_energy = np.nan
                break
            dE_dx = ctxt.get_dE_dx()
            g_norm = mean_norm(dE_dx)
            x_norm = mean_norm(conf)
            # minimization converges when the norm of the forces are less than a certain epsilon
            if g_norm < epsilon:
                break

    # For training purposes, don't raise an exception when energy is nan or minimization doesn't converge
    # Unsuccessful minimizations will have derivatives set to 0 and the data points will not be recorded in the loss plot
    if i == max_iter-1:
        final_energy = np.nan
        print("Energy minimization failed to converge in ", i, "steps")
    else:
        final_energy = ctxt.get_E()
        print("Minimization converged in", i, "steps to", final_energy)


    # IN PROGRESS: dynamics are currently turned off for training
    # # modify integrator to do dynamics
    # opt.set_dt(dt)
    # opt.set_coeff_a(ca)
    # opt.set_coeff_b(cb)
    # opt.set_coeff_c(cc)

    # # dynamics via reservoir sampling
    # k = n_samples # number of samples we want to keep
    # R = []
    # count = 0

    # for count in range(10000):

    #     # closure around R, and ctxt
    #     def get_reservoir_item(step):
    #         E = ctxt.get_E()
    #         dE_dx = ctxt.get_dE_dx()
    #         dx_dp = ctxt.get_dx_dp()
    #         dE_dp = ctxt.get_dE_dp()

    #         # min_dx = np.amin(dx_dp)
    #         # max_dx = np.amax(dx_dp)
    #         # lhs = np.einsum('kl,mkl->m', dE_dx, dx_dp)
    #         # total_dE_dp = lhs + dE_dp

    #         # print(step, total_dE_dp)

    #         x = ctxt.get_x()

    #         # limits = 1e5
    #         # if min_dx < -limits or max_dx > limits:
    #             # raise Exception("Derivatives blew up:", min_dx, max_dx)

    #         return [E, dE_dx, dx_dp, dE_dp, step]

    #     if count % 1000 == 0:
    #         print(count, ctxt.get_E())

    #     if count < k:
    #         R.append(get_reservoir_item(count))
    #     else:
    #         j = random.randint(0, count)
    #         if j < k:
    #             R[j] = get_reservoir_item(count)

    #     ctxt.step()

    # IN PROGRESS: currently not reservoir sampling, just taking the final state
    R = [[
        final_energy,
        ctxt.get_dE_dx(),
        ctxt.get_dx_dp(),
        ctxt.get_dE_dp(),
        ctxt.get_x()
    ]]

    return R
