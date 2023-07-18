# make clash-resistant propagation function
# a few steps of Barker MCMC (possibly unadjusted), followed by Langevin MD
# (justification: may initialize with clashes (handled by Barker), but otherwise prefer inertial Langevin)

from functools import partial

from jax import grad

from timemachine.constants import BOLTZ, DEFAULT_TEMP
from timemachine.md.barker import BarkerProposal
from timemachine.md.states import CoordsVelBox

# from timemachine.md.thermostat.utils import sample_velocities


class RobustPropagator:
    def __init__(
        self, ctxt, U_fxn, n_barker_steps=100, barker_stepsize_nm=0.001, n_md_steps=1000, temperature=DEFAULT_TEMP
    ):
        """Propagate using a sequence of two approximate MCMC moves --
        the first is robust (Barker -- unlikely to blow up even if initialized with clashes / high force magnitude),
        the second is less robust but more accurate and efficient (inertial Langevin)
        """
        self.ctxt = ctxt
        self.U_fxn = U_fxn  # U_fxn(conf, box) -> energy, -grad(U_fxn)(conf, box) -> forces
        self.n_barker_steps = n_barker_steps
        self.barker_stepsize_nm = barker_stepsize_nm
        self.n_md_steps = n_md_steps

        kBT = BOLTZ * temperature  # TODO: read temperature, masses from ctxt.integrator?

        def log_q(x, box):
            return -U_fxn(x, box) / kBT

        self.log_q = log_q

        def grad_log_q(x, box):
            return -grad(log_q)(x, box)

        self.grad_log_q = grad_log_q

    def propagate(self, xvb: CoordsVelBox) -> CoordsVelBox:  # TODO: modify signature to accept lam?
        x, v, box = xvb.coords, xvb.velocities, xvb.box
        barker = BarkerProposal(partial(self.grad_log_q, box=box), self.barker_stepsize_nm)

        # unadjusted Barker
        for _ in range(self.n_barker_steps):
            x = barker.sample(x)
            # TODO: Pending safety fixes in U(x) impl, add optional Metropolis check.
            #  For now, omit Metropolis check.

        # MD
        self.ctxt.set_x_t(x)
        self.ctxt.set_v_t(v)  # TODO: maybe re-initialize velocities?
        self.ctxt.set_box(box)
        self.ctxt.multiple_steps(self.n_md_steps)

        xvb = CoordsVelBox(self.ctxt.get_x_t(), self.ctxt.get_v_t(), self.ctxt.get_box())

        return xvb
