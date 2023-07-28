__all__ = [
    "one_sided_exp",
]

from typing import Any

from jax import numpy as jnp
from jax.scipy.special import logsumexp

Array = Any  # see https://github.com/google/jax/issues/943


def one_sided_exp(delta_us: Array) -> float:
    """exponential averaging

    ref: https://github.com/choderalab/pymbar/blob/15f932a271343e611ed4be2d468c77b1d11cf01f/pymbar/exp.py#L54
    """
    return -logsumexp(-delta_us - jnp.log(len(delta_us)))
