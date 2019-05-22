import numpy as np
import custom_ops

# test in both modes.
from jax.config import config; config.update("jax_enable_x64", True)
import functools

import jax
from timemachine.potentials import bonded

def batch_mult_jvp(grad_fn, x, p, dxdp):
    dpdp = np.eye(p.shape[0])
    def apply_one(dxdp_i, dpdp_i):
        return jax.jvp(
            grad_fn,
            (x, p),
            (dxdp_i, dpdp_i)
        )
    a, b = jax.vmap(apply_one)(dxdp, dpdp)
    return a[0], b

x0 = np.array([
    [1.0, 0.2, 3.3], # H 
    [-0.5,-1.1,-0.9], # C
    [3.4, 5.5, 0.2], # H 
], dtype=np.float64)
params = np.array([10.0, 3.0, 5.5], dtype=np.float64)
param_idxs = np.array([
    [0,1],
    [1,2],
], dtype=np.int32)
bond_idxs = np.array([
    [0,1],
    [1,2]
], dtype=np.int32)

dxdps = np.random.rand(3, 3, 3).astype(np.float64)

d_a, d_b = custom_ops.harmonic_bond_hmp_gpu_r64(
    x0,
    params,
    dxdps,
    bond_idxs,
    param_idxs
)

energy_fn = functools.partial(bonded.harmonic_bond, box=None, param_idxs=param_idxs, bond_idxs=bond_idxs)
grad_fn = jax.grad(energy_fn, argnums=(0,))

a, b = batch_mult_jvp(grad_fn, x0, params, dxdps)

np.testing.assert_almost_equal(d_a, a[0])
np.testing.assert_almost_equal(d_b, b[0])