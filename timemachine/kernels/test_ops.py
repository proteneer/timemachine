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

def test_derivatives(dx_dp):

    hb = custom_ops.HarmonicBond_f64(
        bond_idxs,
        param_idxs
    )

    if np.prod(dx_dp) == 0:
        extra_arg = np.empty(shape=(0,))
    else:
        extra_arg = dx_dp

    test_e, test_de_dx, test_de_dp, test_d2e_dx2 = hb.derivatives(
        x0,
        params,
        dx_dp=extra_arg,
        dp_idxs=np.array([0], dtype=np.int32),
    )

    energy_fn = functools.partial(
        bonded.harmonic_bond,
        box=None,
        param_idxs=param_idxs,
        bond_idxs=bond_idxs
    )

    grad_fn = jax.grad(energy_fn, argnums=(0,))

    ref_de_dx, ref_d2e_dx2 = batch_mult_jvp(grad_fn, x0, params, dx_dp)
    ref_e, ref_de_dp = batch_mult_jvp(energy_fn, x0, params, dx_dp)

    np.testing.assert_almost_equal(test_e, ref_e)
    np.testing.assert_almost_equal(test_de_dp, ref_de_dp)
    np.testing.assert_almost_equal(test_de_dx, ref_de_dx[0])
    np.testing.assert_almost_equal(test_d2e_dx2, ref_d2e_dx2[0])

test_derivatives(np.random.rand(3, 3, 3).astype(np.float64))
test_derivatives(np.zeros(shape=(3, 3, 3)).astype(np.float64))