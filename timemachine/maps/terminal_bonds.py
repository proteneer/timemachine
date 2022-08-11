from dataclasses import dataclass

import numpy as np
from jax import config, jit
from jax import numpy as jnp

config.update("jax_enable_x64", True)


@dataclass
class Interval:
    lower: float
    upper: float

    @property
    def width(self):
        return self.upper - self.lower

    def validate(self):
        assert self.width > 0
        assert self.lower > 0


@jit
def interval_map(x, src_lb, src_ub, dst_lb, dst_ub):
    scale_factor = (dst_ub - dst_lb) / (src_ub - src_lb)

    in_support = (x >= src_lb) * (x <= src_ub)
    mapped = dst_lb + (x - src_lb) * scale_factor

    return jnp.where(in_support, mapped, np.nan)
