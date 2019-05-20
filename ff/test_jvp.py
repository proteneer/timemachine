import functools
import numpy as onp
import jax
import jax.numpy as np

from timemachine.potentials import bonded

# suppose we have a function

param_idxs = np.array([
    [0,1],
    [1,0],
    [1,0],
], dtype=np.int32)

bond_idxs = np.array([
    [0,1],
    [1,2],
    [2,3]
], dtype=np.int32)

energy_fn = functools.partial(bonded.harmonic_bond, box=None, param_idxs=param_idxs, bond_idxs=bond_idxs)

def hvp(x, p, dxdp):
    h_fn = jax.hessian(energy_fn, argnums=(0,))
    mp_fn = jax.jacfwd(
        jax.grad(energy_fn, argnums=(1,)),
        argnums=(0,))
    return np.einsum('ijkl,mkl->mij', h_fn(x, p)[0][0], dxdp) + mp_fn(x, p)[0][0]

x0 = np.array([
    [1.0, 0.2, 3.3], # H 
    [-0.5,-1.1,-0.9], # C
    [3.4, 5.5, 0.2], # H 
    [5.4, 5.5, 0.2], # H 
], dtype=np.float64)

params = np.array([10.0, 3.0], dtype=np.float64)
# dxdp0 = onp.random.rand(2, 4, 3)
dxdp0 = onp.zeros(shape=(2, 4, 3))

# reference hvp:
print(hvp(x0, params, dxdp0))

grad_fn = jax.grad(energy_fn, argnums=(0,))

def mult_jvp(x, p, dxdp):

    dpdp = np.array([1.0, 0.0])
    _, a = jax.jvp(
        grad_fn,
        (x, p,),
        (dxdp[0], dpdp,)
    )

    dpdp = np.array([0.0, 1.0])

    _, b = jax.jvp(
        grad_fn,
        (x, p,),
        (dxdp[1], dpdp,)
    )

    return np.stack([a[0],b[0]])

print(mult_jvp(x0, params, dxdp0))

def batch_mult_jvp(x, p, dxdp):

    dpdp = np.eye(p.shape[0])

    def apply_one(dxdp_i, dpdp_i):
        return jax.jvp(
            grad_fn,
            (x, p),
            (dxdp_i, dpdp_i,)
        )

    _, grads = jax.vmap(apply_one)(dxdp, dpdp)
    return grads[0]

print(batch_mult_jvp(x0, params, dxdp0))
