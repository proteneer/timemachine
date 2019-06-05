import numpy as np
import random

from system import forcefield
from timemachine.lib import custom_ops
from timemachine.integrator import langevin_coefficients


def ensemble_E_and_derivs(triples):
    """
    Compute the derivative of <dE/dtheta> = <dE/dx.dx/dtheta>
    """
    running_sum_derivs = None
    running_sum_E = 0
    n_triples = len(triples)
    for E, dE_dx, dx_dp in triples:
        if running_sum_derivs is None:
            running_sum_derivs = np.zeros_like(dx_dp)    
        # broadcast multiply [1, N, 3] by [P, N, e]
        running_sum_derivs += np.expand_dims(dE_dx, 0)*dx_dp
        running_sum_E += E
    return E/n_triples, running_sum_derivs/n_triples


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
        
    dt = 0.001
    ca, cb, cc = langevin_coefficients(
        temperature=100.0,
        dt=dt,
        friction=75,
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

    # minimization
    # call system converged when the delta is .25 kcal)
    last_E = None
    max_iter = 5000
    for i in range(max_iter):
        ctxt.step()
        if last_E is None:
            last_E = ctxt.get_E()
        elif np.abs(ctxt.get_E() - last_E) < 1.046:
            break
        else:
            last_E = ctxt.get_E()

        if i % 100 == 0:
            print(i, ctxt.get_E())

    if i == max_iter-1:
        raise Exception("Energy minimization failed to converge in ", i, "steps")
    else:
        print("Minimization converged in", i, "steps")

    # modify integrator to do dynamics
    opt.set_dt(dt)
    opt.set_coeff_a(ca)
    opt.set_coeff_b(cb)
    opt.set_coeff_c(cc)

    # dynamics via reservoir sampling

    k = n_samples # number of samples we want to keep
    R = []
    count = 0

    for count in range(n_steps):

        # closure around R, and ctxt
        def get_reservoir_item():
            E = ctxt.get_E()
            dE_dx = ctxt.get_dE_dx()
            dx_dp = ctxt.get_dx_dp()
            min_dx = np.amin(dx_dp)
            max_dx = np.amax(dx_dp)
            limits = 1e3
            if min_dx < -limits or max_dx > limits:
                raise Exception("Derivatives blew up:", min_dx, max_dx)
            return [E, dE_dx, dx_dp]

        if count < k:
            R.append(get_reservoir_item())
        else:
            j = random.randint(0, count)
            if j < k:
                R[j] = get_reservoir_item()

        ctxt.step()
        if count % 400 == 0:
            print(count, ctxt.get_E())

    return R
