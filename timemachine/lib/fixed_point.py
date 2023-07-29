import jax.numpy as jnp

from timemachine.lib import custom_ops


def FIXED_TO_FLOAT(v) -> jnp.float64:
    return jnp.float64(jnp.int64(jnp.uint64(v))) / custom_ops.FIXED_EXPONENT


def FLOAT_TO_FIXED(v) -> jnp.uint64:
    return jnp.uint64(jnp.int64(v * custom_ops.FIXED_EXPONENT))
