import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from timemachine.lib import custom_ops


def fixed_to_float(v: ArrayLike) -> Array:
    """Meant to imitate the logic of timemachine/cpp/src/fixed_point.hpp::FIXED_TO_FLOAT"""
    return jnp.float64(jnp.int64(jnp.uint64(v))) / custom_ops.FIXED_EXPONENT


def float_to_fixed(v: ArrayLike) -> Array:
    """Meant to imitate the logic of timemachine/cpp/src/kernels/k_fixed_point.cuh::FLOAT_TO_FIXED"""
    return jnp.uint64(jnp.int64(v * custom_ops.FIXED_EXPONENT))
