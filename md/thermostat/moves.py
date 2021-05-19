from md.moves import MonteCarloMove
from md.states import CoordsVelBox
from md.utils import run_thermostatted_md

class UnadjustedMDMove(MonteCarloMove):
    def __init__(self, integrator_impl, bound_impls, lam=1.0, n_steps=5):
        self.integrator_impl = integrator_impl
        self.bound_impls = bound_impls
        self.lam = lam
        self.n_steps = n_steps

    def move(self, x: CoordsVelBox):
        x_t, v_t = run_thermostatted_md(
            self.integrator_impl, self.bound_impls, x, self.lam, n_steps=self.n_steps)
        after_nvt = CoordsVelBox(x_t, v_t, x.box.copy())

        self.n_proposed += 1
        self.n_accepted += 1

        return after_nvt