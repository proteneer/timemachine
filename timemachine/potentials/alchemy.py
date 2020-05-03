import jax.numpy as jnp

def linear_rescale(x, p, lamb, fn0, fn1):
    f_lambda = jnp.sin(jnp.pi*lamb/2)
    f_lambda = f_lambda*f_lambda
    return (1-f_lambda)*fn0(x,p,lamb) + f_lambda*fn1(x,p,lamb)