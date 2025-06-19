from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from timemachine.lib import custom_ops
from timemachine.optimized_kernels import OptimizedLangevinIntegrator

# safe to pickle!

# Replace LangevinIntegrator with optimized version
LangevinIntegrator = OptimizedLangevinIntegrator


@dataclass
class LangevinIntegratorLegacy:
    temperature: float
    dt: float
    friction: float
    masses: NDArray[np.float64]
    seed: int

    def impl(self):
        return custom_ops.LangevinIntegrator(self.masses, self.temperature, self.dt, self.friction, self.seed)


@dataclass
class VelocityVerletIntegrator:
    dt: float
    masses: NDArray[np.float64]

    cbs: NDArray[np.float64] = field(init=False)

    def __post_init__(self):
        cb = self.dt / self.masses
        cb *= -1
        self.cbs = cb

    def impl(self):
        return custom_ops.VelocityVerletIntegrator(self.dt, self.cbs)


@dataclass
class MonteCarloBarostat:
    N: int
    pressure: float
    temperature: float
    group_idxs: Any  # TODO: address mixed convention for type of group_idxs
    interval: int
    seed: int
    adaptive_scaling_enabled: bool = True
    initial_volume_scale_factor: Optional[float] = None

    def impl(self, bound_potentials):
        return custom_ops.MonteCarloBarostat(
            self.N,
            self.pressure,
            self.temperature,
            self.group_idxs,
            self.interval,
            bound_potentials,
            self.seed,
            self.adaptive_scaling_enabled,
            self.initial_volume_scale_factor or 0.0,  # 0.0 is a special value meaning "use 1% of initial box volume"
        )
