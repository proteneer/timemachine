import numpy as np
import random

from system import forcefield
from timemachine.lib import custom_ops
from timemachine.integrator import langevin_coefficients

from timemachine import constants

def average_E_and_derivatives(reservoir):
    """
    Compute the average energy and derivatives

    Parameters
    ----------
    reservoir: list of reservoir
        [
            [E, dE_dx, dx_dp, dE_dp],
            [E, dE_dx, dx_dp, dE_dp],
            ...
        ]

    Returns
    -------
    Average energy, analytic total derivative, and thermodynamic gradient

    """
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

        # tensor contract [N,3] with [P, N, 3] and dE_d
        total_dE_dp = np.einsum('kl,mkl->m', dE_dx, dx_dp) + dE_dp
        running_sum_total_derivs += total_dE_dp
        running_sum_E += E


        running_sum_dE_dp += dE_dp
        running_sum_EmultdE_dp += E*dE_dp

    # compute the thermodynamic average:
    # boltz*(<E><dE/dp> - <E.dE/dp>)
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
    n_steps=1000):

    potentials = forcefield.merge_potentials(potentials)
        
    dt = 0.002
    ca, cb, cc = langevin_coefficients(
        temperature=25.0,
        dt=dt,
        friction=50,
        masses=masses
    )

    m_dt, m_ca, m_cb, m_cc = dt, 0.5, cb, np.zeros_like(masses)

    opt = custom_ops.LangevinOptimizer_f64(
        m_dt,
        m_ca,
        m_cb,
        m_cc
    )

    v0 = np.zeros_like(conf)
    dp_idxs = dp_idxs.astype(np.int32)

    ctxt = custom_ops.Context_f64(
        potentials,
        opt,
        params,
        conf, # x0
        v0, # v0
        dp_idxs
    )

    # Minimize the system and carry the gradient over
    # call system converged when the delta is .25 kcal)
    max_iter = 10000
    window_size = 150
    minimization_energies = []
    for i in range(max_iter):
        ctxt.step()
        E = ctxt.get_E()
        minimization_energies.append(E)
        if len(minimization_energies) > window_size:
            window_std = np.std(minimization_energies[-window_size:])
            if window_std < 1.046:
                break
        if i % 1000 == 0:
            print("minimization", i, E)

    if i == max_iter-1:
        raise Exception("Energy minimization failed to converge in ", i, "steps")
    else:
        print("Minimization converged in", i, "steps to", E)

    # #modify integrator to do dynamics
    # opt.set_dt(dt)
    # opt.set_coeff_a(ca)
    # opt.set_coeff_b(cb)
    # opt.set_coeff_c(cc)

    # # dynamics via reservoir sampling
    # k = n_samples # number of samples we want to keep
    # R = []
    # count = 0

    # for count in range(n_steps):

    #     # closure around R, and ctxt
    #     def get_reservoir_item(step):
    #         E = ctxt.get_E()
    #         dE_dx = ctxt.get_dE_dx()
    #         dx_dp = ctxt.get_dx_dp()
    #         dE_dp = ctxt.get_dE_dp()
    #         min_dx = np.amin(dx_dp)
    #         max_dx = np.amax(dx_dp)
    #         lhs = np.einsum('kl,mkl->m', dE_dx, dx_dp)
    #         total_dE_dp = lhs + dE_dp

    #         # print(step, total_dE_dp)

    #         limits = 1e5
    #         # if min_dx < -limits or max_dx > limits:
    #             # raise Exception("Derivatives blew up:", min_dx, max_dx)
    #         return [E, dE_dx, dx_dp, dE_dp, step]

    #     if count < k:
    #         R.append(get_reservoir_item(count))
    #     else:
    #         j = random.randint(0, count)
    #         if j < k:
    #             R[j] = get_reservoir_item(count)
    #             np.set_printoptions(suppress=True)

    #     if count % 5000 == 0:
    #         print("count", count)

    #     ctxt.step()

    R = [[
        ctxt.get_E(),
        ctxt.get_dE_dx(),
        ctxt.get_dx_dp(),
        ctxt.get_dE_dp(),
        0
    ]]

    return R
