from timemachine.lib import custom_ops
from timemachine.md.moves import MonteCarloMove
from timemachine.md.states import CoordsVelBox


class UnadjustedLangevinMove(MonteCarloMove):
    def __init__(self, integrator_impl, bound_impls, n_steps=5):
        self.integrator_impl = integrator_impl
        self.bound_impls = bound_impls
        self.n_steps = n_steps

    def move(self, x: CoordsVelBox):
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

        self.n_proposed += 1
        self.n_accepted += 1

        return after_nvt
