from timemachine.potentials import bonded
from jax.config import config; config.update("jax_enable_x64", True)
import numpy as np
import functools
import jax
dt = np.array(0.002)
ca = np.array(0.9)
# ca = np.array(0.99)
# ca = np.array(0.05)
# ca = 0.05 # this destroys information
cb = -np.array(0.001)
masses = np.array([1.0, 12.0, 4.0, 3.0])
x0 = np.array([
    [1.0, 0.5, -0.5],
    [0.2, 0.1, -0.3],
    [0.5, 0.4, 0.3],
    [0.8, 0.3, 0.4],
], dtype=np.float64)
x0.setflags(write=False)

num_atoms = x0.shape[0]

params = np.array([25.0, 2.0, 75.0, 1.81, 5.0], np.float64)

bond_idxs = np.array([[0, 1], [1, 2], [0, 3]], dtype=np.int32)
bond_param_idxs = np.array([[0, 1], [0, 1], [1, 0]], dtype=np.int32)

angle_idxs = np.array([[0,1,2]], dtype=np.int32)
angle_param_idxs = np.array([[2,3]], dtype=np.int32)

# 1. Reference integration.
ref_hb = functools.partial(bonded.harmonic_bond,
    bond_idxs=bond_idxs,
    param_idxs=bond_param_idxs,
    box=None
)

ref_ha = functools.partial(bonded.harmonic_angle,
    angle_idxs=angle_idxs,
    param_idxs=angle_param_idxs,
    box=None
)

def total_nrg(conf, params):
    return ref_hb(conf, params) + ref_ha(conf, params)

grad_fn = jax.jit(jax.grad(total_nrg, argnums=(0,)))
steps = 1

def r2i(a, scale=1e10):
    return (scale*a).astype(np.int64)

def i2r(a, scale=1e10):
    return a.astype(np.float64)/scale


def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m

// let ca == 

ca_10 = r2i(ca,10) # int
inv_ca_10 = np.array(modinv(ca_10, 2**64)).astype(np.int64) # int

def integrate_fwd(x_t, v_t):
    assert x_t.dtype == np.int64
    assert v_t.dtype == np.int64
    for s in range(steps):
        # v_t, x_t both in 1e10 exponents
        lhs = ca_10*v_t # we can compute an inverse of r2i(ca, 1e1)
        print("lhs", lhs)
        lhs = i2r(lhs,1e11)
        lhs = r2i(lhs)
        v_t = lhs + r2i(cb*grad_fn(i2r(x_t), params)[0])
        x_t = x_t + r2i(i2r(v_t)*dt)

    return x_t, v_t

def integrate_fwd_real(x_t, v_t):
    for s in range(steps):
        lhs = ca*v_t
        v_t = lhs + cb*grad_fn(x_t, params)[0]
        x_t = x_t + v_t*dt
    return x_t, v_t    

def integrate_bwd(x_t, v_t):
    assert x_t.dtype == np.int64
    assert v_t.dtype == np.int64
    for s in range(steps):
        x_t = x_t - r2i(i2r(v_t)*dt)
        v_t = v_t - r2i(cb*grad_fn(i2r(x_t), params)[0]) # int64
        v_t = i2r(v_t)
        v_t = r2i(v_t,1e11)
        v_t = v_t*inv_ca_10

    return x_t, v_t



start = np.array(np.random.rand()).astype(np.float32)
lhs = r2i(start, 2**32)
lhs = i2r(lhs, 2**32)
print(start, lhs)
assert start == lhs




# v0 = np.random.rand(x0.shape[0], x0.shape[1])

# print(x0,v0)

# xf, vf = integrate_fwd_real(x0, v0)
# print("original", xf, vf)
# xf, vf = integrate_fwd(r2i(x0), r2i(v0))
# print("fixed", i2r(xf), i2r(vf))
# xo, vo = integrate_bwd(xf, vf)

# print("x_v, v_f", i2r(xf), i2r(vf))
# print("x_o, v_o", i2r(xo),i2r(vo))