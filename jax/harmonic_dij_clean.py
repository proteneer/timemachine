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

# min_dij = np.array(999.9)
# max_dij = np.array(0.0)

n_smaller = 0
n_bigger = 0

def analytic_grad(coords, params):
    kb = params[0]
    b0 = params[1]

    src_idxs = [0]
    dst_idxs = [1]

    ci = coords[src_idxs]
    cj = coords[dst_idxs]

    dx = ci - cj
    dij = np.linalg.norm(dx, axis=1)
    db = dij - b0

    # global min_dij
    # global max_dij

    global n_bigger
    global n_smaller

    if dij > 0.129:
        n_bigger += 1
        print("BIGGER", n_bigger, dij)
    else:
        n_smaller += 1
        print("SMALLER", n_smaller, dij)

    # if dij < min_dij:
    #     min_dij = dij

    # if dij > max_dij:
    #     max_dij = dij

    # print("dij", dij)

    lhs = kb*db*dx/dij

    src_grad = lhs
    dst_grad = -src_grad
    res = np.concatenate([src_grad, dst_grad], axis=0)

    return res


def langevin_integrator(params, dt=0.0015, friction=1.0, temp=170.0):

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

    for step in range(2000):
        # if step > 100:
            # break

        g = agj(x0, params)

        # perfect NVE
        # vscale = 0.0
        # nscale = 1.0

        # with NVE the period is about once every 3-4 steps

        # with NVT there is no period

        noise = vnp.random.normal(size=(num_atoms, num_dims)).astype(x0.dtype)

        # print("fscale", fscale, "invMasses", invMasses)

        # print("x0", x0)
        # print("A")
        # print_tangent(vscale*v_t)
        # print("B")
        # print_tangent(fscale*invMasses*g) 
        # print("C")
        # print(nscale*sqrtInvMasses*noise) # derivative is zero! remember

        # print("delta:")
        print_tangent(fscale*invMasses*g)
        v_t = vscale*v_t - fscale*invMasses*g + nscale*sqrtInvMasses*noise
        # print("C")
        # print_tangent(v_t)

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

    theta = np.array([2500.0, 0.129], dtype=np.float64)

    # grads = np.array([1.0, 0.0], dtype=np.float64)

    # primals, tangents = jax.jvp(langevin_integrator, (theta,), (np.ones_like(theta),))

    # print("primals", primals)
    # print("tangents", tangents)

    grads = np.array([0.0, 1.0], dtype=np.float64)

    primals, tangents = jax.jvp(langevin_integrator, (theta,), (np.ones_like(theta),))

    print("primals", primals)
    print("tangents", tangents)

    assert 0

    res = dxdp(x, theta)[0]


    print("min", min_dij, "max", max_dij)
    # print("all_dijs", vnp.amin(all_dijs))
    print(res, np.amax(res), np.amin(res))