import time
import numpy as vnp
from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np


from jax import custom_transforms
from jax.interpreters.ad import defjvp

def print_tangent_jvp(t, x):
  print(t)
  return t

print_tangent = custom_transforms(lambda x: x)
defjvp(print_tangent.primitive, print_tangent_jvp)

import scipy.stats as stats

BOLTZMANN = 1.380658e-23
AVOGADRO = 6.0221367e23
RGAS = BOLTZMANN*AVOGADRO
BOLTZ = RGAS/1000
ONE_4PI_EPS0 = 138.935456
VIBRATIONAL_CONSTANT = 1302.79 # http://openmopac.net/manual/Hessian_Matrix.html

min_dij = np.array(999.9)
max_dij = np.array(0.0)



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

    # global min_dij

    # print(dij)
    # print(dir(dij))
    # print(vnp.asarray(dij.aval))
    # print(vnp.array(dij.aval))
    # f

    # if dij < min_dij:
        # min_dij = dij

    # print("DIJ", dij)
    # energy = np.sum(kb*np.power(dij - b0, 2)/2)
    # energy = -kb*np.exp(-np.abs(dij-b0))
    energy = kb*np.power(dij-b0, 2)/2

    print("energy", energy)

    return np.sum(energy)


def harmonic_bond_grad(coords, params):
    return jax.jacrev(harmonic_bond_nrg, argnums=(0,))

def analytic_grad(coords, params):
    kb = params[0]
    kb = 250000
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

    global min_dij
    global max_dij

    if dij < min_dij:
        min_dij = dij

    if dij > max_dij:
        max_dij = dij


#     print("dij---", dir(dij))
#     print("dij---", dij.pval)
#     print("dij---", dij.aval.astype)
#     assert 0

# # min_dij:
#         # print(dij.aval)

#     # print("?",  vnp.array(dij.aval).tolist()) 
#     all_dijs.append(dij.aval)


    lhs = kb*db*dx/dij

    # print(lhs.shape)
    # assert 0

    src_grad = lhs
    dst_grad = -src_grad

    # dx0 = np.sum(src_grad, axis=0, keepdims=True)
    res = np.concatenate([src_grad, dst_grad], axis=0)

    return res

def nose_hoover_integrator(x0, params, dt=0.0025, friction=1.0, temp=300.0):

    masses = np.array([1.0, 1.0], dtype=np.float64)
    masses = masses.reshape((-1, 1))

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
    invMasses = (1.0/masses)
    sqrtInvMasses = np.sqrt(invMasses)

    coeff_a = vscale
    coeff_bs = fscale*invMasses
    coeff_cs = nscale*sqrtInvMasses

    start_time = time.time()

    # agj = jax.jit(analytic_grad)
    agj = analytic_grad
    r_t = x0
    z_t = 0
    f_t = -agj(r_t, params)

    Q = friction # or vscale?
    Q = vscale

    for step in range(5000):

        # f_t = -g
        r_dt = r_t + v_t*dt + (f_t*invMasses - z_t*v_t)*dt*dt/2
        v_dt_2 = v_t + (dt/2)*(f_t*invMasses - z_t*v_t)
        f_dt = -agj(r_dt, params)

        KE_dt = np.sum(0.5*v_t*v_t*masses)
        KE_dt_2 = np.sum(0.5*v_dt_2*v_dt_2*masses)

        z_dt_2 = z_t + (dt/(2*Q))*(KE_dt-kT*(3*num_atoms+1)/2)
        z_dt = z_dt_2 + (dt/(2*Q))*(KE_dt_2-kT*(3*num_atoms+1)/2)
        v_dt = (v_dt_2+(dt/2)*f_dt*invMasses)/(1+(dt/2)*z_dt)

        v_t = v_dt
        r_t = r_dt
        f_t = f_dt
        z_t = z_dt

        PE = harmonic_bond_nrg(x0, params)
        # KE = np.sum(0.5*v_t*v_t/invMasses)
        TE = (PE + KE_dt)
        KE = TE - PE

        # print(dir(KE_dt), KE_dt.__array__())

        # assert 0/

        print(step, "NH speed", (time.time() - start_time)/(step+1), np.amax(v_t).aval, "TE", TE.aval, "KE", KE.aval, "Z_T", z_t.aval)

    return r_t



def langevin_integrator(params, dt=0.0025, friction=1.0, temp=300.0):

    # print("params", params)
    # assert 0

    x0 = np.array([
        [-0.0036,  0.0222,  0.0912],
        [-0.0162, -0.8092,  0.7960],
        # [-0.1092,  0.9610,  0.6348],
        # [-0.8292, -0.0852, -0.6123]
    ], dtype=np.float64)

    x0 = x0/10

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

    agj = analytic_grad

    KEs = []

    max_PE = 0

    for step in range(10000):
        # func = harmonic_bond_grad(x0, params)
        # g = func(x0, params)[0]

        g = agj(x0, params)

        noise = vnp.random.normal(size=(num_atoms, num_dims)).astype(x0.dtype)
        # nscale = 0.0 # NVE

        v_t = vscale*v_t - fscale*invMasses*g + nscale*sqrtInvMasses*noise

        # dv_t/params

        dx = v_t * dt
        # PE = harmonic_bond_nrg(x0, params)

        # if PE > max_PE:
            # max_PE = PE

        # KE = np.sum(0.5*v_t*v_t/invMasses)
        # TE = (PE + KE).aval

        # print("min_dij", min_dij)

        # print(step, "speed", (time.time() - start_time)/(step+1), np.amax(v_t).aval, "TE", TE, "PE", PE)
        print(step, "speed")
        x0 += dx

    

    print("MAX_PE", max_PE)
    print(coeff_a, coeff_bs, coeff_cs)


    return x0

if __name__ == "__main__":


    theta = np.array([25000.0, 0.129], dtype=np.float64)
    # theta = np.array([250000.0, 0.120], dtype=np.float64)


    # a = harmonic_bond_grad(x, theta)(x, theta)[0]
    # b = analytic_grad(x, theta)

    # print(a - b)
    # assert np.max(a-b) < 1e-7

    # assert 0

    # dxdp = jax.jacfwd(langevin_integrator, argnums=(1,))


    # it's the mixed partials that's a lot more unstable than the hessian...?
    primals, tangents = jax.jvp(langevin_integrator, (theta,), (np.ones_like(theta),))

    print(primals, tangents)

    assert 0

    res = dxdp(x, theta)[0]


    print("min", min_dij, "max", max_dij)
    # print("all_dijs", vnp.amin(all_dijs))
    print(res, np.amax(res), np.amin(res))