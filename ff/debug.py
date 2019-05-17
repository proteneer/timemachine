import jax
import jax.numpy as np
import numpy as onp

def fn(a,b):
    return a*a*b*b

grad_fn = jax.grad(fn, argnums=(0, 1))
print(onp.asarray(grad_fn(1.0,5.0)))
