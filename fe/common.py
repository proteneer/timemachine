import time
import numpy as np

from system import forcefield

import jax

from timemachine.lib import custom_ops
from timemachine.integrator import langevin_coefficients

def dU_dlambda(dE_dx, nha):
    """
    Compute the 4 dimensional dU/dlambda via decoupling.
    """
    return np.sum(dE_dx[nha:, 3:]) 

def ns_per_day(delta, timestep_in_fs):
    return (timestep_in_fs/delta)*(86400)*1e-6

def compute_d2u_dldp(energies, params, xs, dx_dps, dp_idxs, num_host_atoms):

    assert len(xs.shape) == 2
    assert len(dx_dps.shape) == 3

    mixed_partials = []
    hessians = []
    # we need to compute this separately since the context's sgemm call overwrites
    # the values of d2u_dxdp
    # batched call
    for p in energies:
        _, _, ph, _, pmp  = p.derivatives(np.expand_dims(xs, axis=0), params, dp_idxs)
        mixed_partials.append(pmp)
        hessians.append(ph)
    
    hessians = np.sum(hessians, axis=0)[0]
    mixed_part = np.sum(mixed_partials, axis=0)[0]

    hess_idxs = jax.ops.index[num_host_atoms:, 3:, :, :3]
    dx_dp_idxs = jax.ops.index[:, :, :3]
    mp_idxs = jax.ops.index[:, num_host_atoms:, 3:]
    lhs = np.einsum('ijkl,mkl->mij', hessians[hess_idxs], dx_dps[dx_dp_idxs]) # correct only up to main hessian
    rhs = mixed_part[mp_idxs]

    # lhs + rhs has shape [P, num_atoms-num_host_atoms, 1] 
    d2u_dldp = np.sum(lhs+rhs, axis=(1,2)) # P N 4 -> P
    return d2u_dldp


def minimize(
    num_host_atoms,
    potentials,
    params,
    conf,
    masses,
    dp_idxs,
    starting_dimension,
    lamb):
    """
    Minimize a structure.
    """

    num_atoms = len(masses)
    potentials = forcefield.merge_potentials(potentials)

    dt = 1e-3
    ca, cb, cc = langevin_coefficients(
        temperature=300,
        dt=dt,
        friction=100, # (ytz) probably need to double this?
        masses=np.ones_like(masses)*10,
    )

    m_dt, m_ca, m_cb, m_cc = dt, 0.0, np.ones_like(cb)/10000, np.zeros_like(masses)

    opt = custom_ops.LangevinOptimizer_f64(
        dt,
        4,
        m_ca,
        m_cb.astype(np.float64),
        m_cc.astype(np.float64)
    )

    dp_idxs = dp_idxs.astype(np.int32)

    count = 0

    # prod
    # max_iter = 40000 # of steps for minimization
    # dt = 1e-9

    # debug
    max_iter = 4000 # of steps for minimization
    dt = 1e-5
    increment = 1.005
    max_dt = 0.007

    num_atoms = conf.shape[0]
    num_dimensions = starting_dimension
    
    d4_t = np.zeros((num_atoms, num_dimensions), dtype=np.float64)
    d4_t_lambdas = np.zeros((num_atoms, num_dimensions), dtype=np.float64) + lamb

    # set coordinates
    d4_t[:num_host_atoms, :3] = conf[:num_host_atoms, :3]
    d4_t[num_host_atoms:, :3] = conf[num_host_atoms:, :3]
    d4_t[num_host_atoms:, 3:] = d4_t_lambdas[num_host_atoms:, 3:]

    x_t = d4_t
    v_t = np.zeros_like(x_t)

    ctxt = custom_ops.Context_f64(
        potentials,
        opt,
        params.astype(np.float64),
        x_t.astype(np.float64),
        v_t.astype(np.float64), # n
        dp_idxs.astype(np.int32)
    )

    start_time = time.time()



    all_xis = []


    for i in range(max_iter):
        dt *= increment
        dt = min(dt, max_dt)

        opt.set_dt(dt)
        ctxt.step()

        if i % 200 == 0 or i == max_iter-1:
            E = ctxt.get_E()
            xi = ctxt.get_x()
            dE_dx = ctxt.get_dE_dx()
            dU_dl = dU_dlambda(dE_dx, num_host_atoms)
            dx_dp = ctxt.get_dx_dp()

            speed = ns_per_day(time.time()-start_time, i)
            print("step", i, "lambda", lamb, "dt", dt, "Energy", E,  "min/max/mean dxdp", np.amin(dx_dp), np.amax(dx_dp), np.mean(dx_dp), "du/dl", dU_dl, "speed", speed, "ns/day")

            if np.isnan(E):
                assert 0

            all_xis.append(xi)

    # Append final du/dl
    all_dudls = []
    all_dudls.append(dU_dl)

    all_d2u_dldps = []
    # Compute the corresponding derivative
    all_d2u_dldps.append(compute_d2u_dldp(
        potentials,
        params.astype(np.float64),
        xi, # final x_i
        dx_dp, # final dx_dp
        dp_idxs, # final dp_idxs
        num_host_atoms)
    )

    return all_dudls, all_d2u_dldps, all_xis