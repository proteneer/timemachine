import jax.numpy as jnp
import matplotlib.pyplot as plt

def switch_fn(dij, cutoff):
    """assumes cutoff=1.0"""
    return jnp.power(jnp.cos((jnp.pi * jnp.power(dij, 8)) / (2 * cutoff)), 2)

def switch_fn_2(dij, cutoff):
    """possible normalization of switch_fn, but untested -- still might introduce large forces near cutoff"""
    return jnp.power(jnp.cos((jnp.pi * jnp.power(dij/cutoff, 8)) / 2), 2)

cutoff = 1.2
dij = jnp.linspace(0, cutoff, 10000)
plt.plot(dij, switch_fn(dij, cutoff), label='switch_fn');
plt.plot(dij, switch_fn_2(dij, cutoff), label='switch_fn_2');
plt.xlabel('dij'); plt.ylabel('f(dij)'); plt.legend();
