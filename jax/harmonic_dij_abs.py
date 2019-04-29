import time
import numpy as vnp
from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np


import scipy.stats as stats

BOLTZMANN = 1.380658e-23
AVOGADRO = 6.0221367e23
RGAS = BOLTZMANN*AVOGADRO
BOLTZ = RGAS/1000
ONE_4PI_EPS0 = 138.935456
VIBRATIONAL_CONSTANT = 1302.79 # http://openmopac.net/manual/Hessian_Matrix.html

def harmonic_bond_nrg(
        coords,
        params):
    kb = params[0]
    b0 = params[1]

    src_idxs = [0]
    dst_idxs = [1]

    ci = coords[src_idxs]
    cj = coords[dst_idxs]

    dx = ci - cj
    dij = np.linalg.norm(dx, axis=1)

    # print("DIJ", dij)
    # energy = np.sum(kb*np.power(dij - b0, 2)/2)
    # energy = -kb*np.exp(-np.abs(dij-b0))
    energy = kb*np.abs(dij-b0)

    print("energy", energy)

    return np.sum(energy)


def harmonic_bond_grad(coords, params):
    return jax.jacrev(harmonic_bond_nrg, argnums=(0,))

def analytic_grad(coords, params):
    kb = params[0]
    b0 = params[1]

    # src_idxs = [0, 0, 0, 0]
    # dst_idxs = [1, 2, 3, 4]

    src_idxs = [0]
    dst_idxs = [1]

    ci = coords[src_idxs]
    cj = coords[dst_idxs]

    dx = ci - cj
    dij = np.linalg.norm(dx, axis=1)
    db = dij - b0

    # lhs = np.expand_dims((db/np.abs(db))*kb*np.exp(-np.abs(db))/dij, axis=-1)
    # rhs = dx

    lhs = kb*((dij-b0)/np.abs(dij-b0))*dx/dij

    # print(lhs.shape)
    # assert 0

    src_grad = lhs
    dst_grad = -src_grad

    # dx0 = np.sum(src_grad, axis=0, keepdims=True)
    res = np.concatenate([src_grad, dst_grad], axis=0)

    return res

# def nose_hoover_integrator(x0, params, dt=0.0025, friction=1.0, temp=300.0):

#     masses = np.array([12.0107, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
#     masses = masses.reshape((-1, 1))

#     num_atoms = len(masses)
#     num_dims = 3

#     dt = dt
#     v_t = np.zeros((num_atoms, num_dims))

#     friction = friction # dissipation speed (how fast we forget)
#     temperature = temp           # temperature

#     vscale = np.exp(-dt*friction)

#     if friction == 0:
#         fscale = dt
#     else:
#         fscale = (1-vscale)/friction
#     kT = BOLTZ * temperature
#     nscale = np.sqrt(kT*(1-vscale*vscale)) # noise scale
#     # normal = tf.distributions.Normal(loc=0.0, scale=1.0)
#     invMasses = (1.0/masses)
#     sqrtInvMasses = np.sqrt(invMasses)

#     coeff_a = vscale
#     coeff_bs = fscale*invMasses
#     coeff_cs = nscale*sqrtInvMasses

#     start_time = time.time()

#     agj = jax.jit(analytic_grad)
#     r_t = x0
#     z_t = 0
#     f_t = -agj(r_t, params)

#     Q = friction # or vscale?
#     Q = vscale

#     for step in range(5000):

#         # f_t = -g
#         r_dt = r_t + v_t*dt + (f_t*invMasses - z_t*v_t)*dt*dt/2
#         v_dt_2 = v_t + (dt/2)*(f_t*invMasses - z_t*v_t)
#         f_dt = -agj(r_dt, params)

#         KE_dt = np.sum(0.5*v_t*v_t*masses)
#         KE_dt_2 = np.sum(0.5*v_dt_2*v_dt_2*masses)

#         z_dt_2 = z_t + (dt/(2*Q))*(KE_dt-kT*(3*num_atoms+1)/2)
#         z_dt = z_dt_2 + (dt/(2*Q))*(KE_dt_2-kT*(3*num_atoms+1)/2)
#         v_dt = (v_dt_2+(dt/2)*f_dt*invMasses)/(1+(dt/2)*z_dt)

#         v_t = v_dt
#         r_t = r_dt
#         f_t = f_dt
#         z_t = z_dt

#         PE = harmonic_bond_nrg(x0, params)
#         # KE = np.sum(0.5*v_t*v_t/invMasses)
#         TE = (PE + KE_dt)
#         KE = TE - PE

#         # print(dir(KE_dt), KE_dt.__array__())

#         # assert 0/

#         print(step, "NH speed", (time.time() - start_time)/(step+1), np.amax(v_t).aval, "TE", TE.aval, "KE", KE.aval, "Z_T", z_t.aval)

#     return r_t

def langevin_integrator(x0, params, dt=0.002, friction=1.0, temp=300.0):

    # masses = np.array([12.0107, 1.0], dtype=np.float64)
    masses = np.array([1.0, 1.0], dtype=np.float64)

    num_atoms = len(masses)
    num_dims = 3

    dt = dt
    v_t = np.zeros((num_atoms, num_dims))

    friction = friction # dissipation speed (how fast we forget)
    temperature = temp           # temperature

    vscale = np.exp(-dt*friction)

    if friction == 0:
        fscale = dt
    else:
        fscale = (1-vscale)/friction
    kT = BOLTZ * temperature
    nscale = np.sqrt(kT*(1-vscale*vscale)) # noise scale
    # normal = tf.distributions.Normal(loc=0.0, scale=1.0)
    invMasses = (1.0/masses).reshape((-1, 1))
    sqrtInvMasses = np.sqrt(invMasses)

    coeff_a = vscale
    coeff_bs = fscale*invMasses
    coeff_cs = nscale*sqrtInvMasses

    start_time = time.time()

    # agj = jax.jit(harmonic_bond_grad)

    agj = jax.jit(analytic_grad)

    KEs = []

    max_PE = 0

    for step in range(1000):
        # func = harmonic_bond_grad(x0, params)
        # g = func(x0, params)[0]
        g = agj(x0, params)
        # random normal
        noise = vnp.random.normal(size=(num_atoms, num_dims)).astype(x0.dtype)

        # vscale = 0.0
        # nscale = 0.0 # NVE
        v_t = vscale*v_t - fscale*invMasses*g + nscale*sqrtInvMasses*noise

        # print(v_t*dt)


        print("X0", x0)
        dx = v_t * dt
        PE = harmonic_bond_nrg(x0, params)

        if PE > max_PE:
            max_PE = PE

        KE = np.sum(0.5*v_t*v_t/invMasses)
        TE = (PE + KE).aval

        print(step, "speed", (time.time() - start_time)/(step+1), np.amax(v_t).aval, "TE", TE, "PE", PE)
        x0 += dx

    print("MAX_PE", max_PE)
    print(coeff_a, coeff_bs, coeff_cs)

    return x0

if __name__ == "__main__":

    x = np.array([
        [-0.0036,  0.0222,  0.0912],
        [-0.0162, -0.8092,  0.7960],
        # [-0.1092,  0.9610,  0.6348],
        # [-0.8292, -0.0852, -0.6123]
    ], dtype=np.float64)

    x = x/10;

    theta = np.array([2500.0, 0.129], dtype=np.float64)
    # theta = np.array([250000.0, 0.120], dtype=np.float64)


    a = harmonic_bond_grad(x, theta)(x, theta)[0]
    b = analytic_grad(x, theta)

    print(a - b)
    assert np.max(a-b) < 1e-7

    # assert 0

    dxdp = jax.jacfwd(langevin_integrator, argnums=(1,))
    res = dxdp(x, theta)[0]
    print(res, np.amax(res), np.amin(res))