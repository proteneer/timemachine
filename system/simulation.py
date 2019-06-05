import numpy as np

from system import forcefield
from timemachine.lib import custom_ops
from timemachine.integrator import langevin_coefficients

def run_simulation(
    potentials,
    params,
    param_groups,
    conf,
    masses,
    dp_idxs):

    potentials = forcefield.merge_potentials(potentials)
        
    dt = 0.000001
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
    for i in range(3000):
        ctxt.step()
        if i % 100 == 0:
            print(i, ctxt.get_E())

    # modify integrator to do dynamics
    opt.set_dt(dt)
    opt.set_coeff_a(ca)
    opt.set_coeff_b(cb)
    opt.set_coeff_c(cc)

    # dynamics
    for i in range(10000):
        if i % 100 == 0:
            print(i, ctxt.get_E())
        ctxt.step()
    


    return ctxt.get_x(), ctxt.get_dx_dp()
