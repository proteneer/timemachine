from md.moves import MonteCarloMove
from md.states import CoordsVelBox, get_coords_vel_box, set_coords_vel_box
from timemachine.lib import custom_ops
import numpy as np


class UnadjustedLangevinMove(MonteCarloMove):
    def __init__(self, integrator_impl, bound_impls, lam=1.0, n_steps=5):
        self.integrator_impl = integrator_impl
        self.bound_impls = bound_impls
        self.lam = lam
        self.n_steps = n_steps
        self.ctxt = None

    def move(self, x: CoordsVelBox):
        if self.ctxt is None:
            # note: context creation overhead here is actually very small!
            self.ctxt = custom_ops.Context(
                x.coords,
                x.velocities,
                x.box,
                self.integrator_impl,
                self.bound_impls,
            )
        else:
            set_coords_vel_box(self.ctxt, x)

        # arguments: lambda_schedule, du_dl_interval, x_interval
        _ = self.ctxt.multiple_steps(self.lam * np.ones(self.n_steps), 0, 0)
        after_nvt = get_coords_vel_box(self.ctxt)

        self.n_proposed += 1
        self.n_accepted += 1

        return after_nvt
