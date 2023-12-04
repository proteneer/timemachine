from jax import numpy as jnp


def switch_fn(dij, cutoff):
    return jnp.power(jnp.cos((jnp.pi * jnp.power(dij, 8)) / (2 * cutoff)), 2)
