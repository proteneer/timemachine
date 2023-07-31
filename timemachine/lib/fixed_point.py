import jax.numpy as jnp

from timemachine.lib import custom_ops


def fixed_to_float(v: int | jnp.uint64) -> jnp.float64:
    """Meant to imitate the logic of timemachine/cpp/src/fixed_point.hpp::FIXED_TO_FLOAT"""
    return jnp.float64(jnp.int64(jnp.uint64(v))) / custom_ops.FIXED_EXPONENT


def float_to_fixed(v: jnp.float32 | float) -> jnp.uint64:
    """Meant to imitate the logic of timemachine/cpp/src/kernels/k_fixed_point.cuh::FLOAT_TO_FIXED"""
    return jnp.uint64(jnp.int64(v * custom_ops.FIXED_EXPONENT))
