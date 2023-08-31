from typing import Tuple

from jax.random import PRNGKeyArray

from timemachine.lib import custom_ops
from timemachine.md.moves import Move, random_seed
from timemachine.md.states import CoordsVelBox


class UnadjustedLangevinMove(Move):
    def __init__(self, integrator_impl: custom_ops.LangevinIntegrator, bound_impls, n_steps=5):
        self.integrator_impl = integrator_impl
        self.bound_impls = bound_impls
        self.n_steps = n_steps

    def move(self, key: PRNGKeyArray, x: CoordsVelBox) -> Tuple[PRNGKeyArray, CoordsVelBox]:
        key, seed = random_seed(key)
        self.integrator_impl.set_seed(seed)

        # note: context creation overhead here is actually very small!
        ctxt = custom_ops.Context(
            x.coords,
            x.velocities,
            x.box,
            self.integrator_impl,
            self.bound_impls,
        )

        # arguments: lambda_schedule, du_dl_interval, x_interval
        _ = ctxt.multiple_steps(self.n_steps, 0)
        x_t = ctxt.get_x_t()
        v_t = ctxt.get_v_t()

        after_nvt = CoordsVelBox(x_t, v_t, x.box.copy())

        return key, after_nvt
