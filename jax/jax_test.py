# def harmonic_force(coords, idxs, kb, ka):
import numpy as vnp
import jax
import jax.numpy as np

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

    src_idxs = [0, 0, 0, 0]
    dst_idxs = [1, 2, 3, 4]

    ci = coords[src_idxs]
    cj = coords[dst_idxs]

    dx = ci - cj
    dij = np.linalg.norm(dx, axis=1)


    # print("dij", dij, dij-b0)
    energy = np.sum(kb*np.power(dij - b0, 2)/2)

    print("nrg", energy)

    return energy


def harmonic_bond_grad(coords, params):
    return jax.jacrev(harmonic_bond_nrg, argnums=(0,))

def analytic_grad(coords, params):
    kb = params[0]
    b0 = params[1]

    src_idxs = [0, 0, 0, 0]
    dst_idxs = [1, 2, 3, 4]

    ci = coords[src_idxs]
    cj = coords[dst_idxs]

    dx = ci - cj
    dij = np.linalg.norm(dx, axis=1)
    db = dij - b0

    lhs = np.expand_dims(kb*db/dij, axis=-1)
    rhs = dx
    src_grad = lhs * rhs
    dst_grad = -src_grad

    dx0 = np.sum(src_grad, axis=0, keepdims=True)
    res = np.concatenate([dx0, dst_grad], axis=0)

    return res

def integrator(x0, params, dt=0.0025, friction=1.0, temp=300.0):

    masses = np.array([12.0107, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

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


    for time in range(1000):
        # func = harmonic_bond_grad(x0, params)
        # g = func(x0, params)[0]
        g = analytic_grad(x0, params)


        noise = vnp.random.normal(size=(num_atoms, num_dims)).astype(x0.dtype)
        v_t = vscale*v_t - fscale*invMasses*g + nscale*sqrtInvMasses*noise
        dx = v_t * dt
        print(time, x0)
        x0 += dx

    return x0

if __name__ == "__main__":

    x = np.array([
        [-0.0036,  0.0222,  0.0912],
        [-0.0162, -0.8092,  0.7960],
        [ 0.9404,  0.0222, -0.4538],
        [-0.1092,  0.9610,  0.6348],
        [-0.8292, -0.0852, -0.6123]
    ], dtype=np.float64)

    theta = np.array([10.0, 1.15], dtype=np.float64)


    a = harmonic_bond_grad(x, theta)(x, theta)[0]
    b = analytic_grad(x, theta)
    assert np.max(a-b) < 1e-7

    dxdp = jax.jacfwd(integrator, argnums=(1,))
    print(dxdp(x, theta))