from jax import config

config.update("jax_enable_x64", True)

from jax import grad, numpy as np
from jax import custom_jvp

from collections import OrderedDict


def stringify_ubp(ubp):
    return ubp.__class__.__name__


class UnboundPotentialEnergyModel:
    def __init__(self, topology, ff, unbound_potentials, precision=np.float64, box=np.eye(3) * 100):
        self.topology = topology
        self.ff = ff
        self.precision = precision
        self.unbound_potentials = unbound_potentials
        self.unbound_impls = self._get_impls(unbound_potentials)
        self.box = box

    def _get_impls(self, unbound_potentials):
        return [ubp.unbound_impl(self.precision) for ubp in unbound_potentials]

    @property
    def all_impls(self):
        """List of impl, e.g. as required by context constructor"""
        return self.unbound_impls

    def handle_optional_box(self, box=None):
        if box is None:
            return self.box
        else:
            return box

    def apply_params(self, ff_params):
        raise(NotImplementedError)


    def execute_U(self, x, lam, ff_params, box=None):
        """TODO: reduce code duplication between execute_U, execute_dU_dx, execute_dU_dlam"""
        box = self.handle_optional_box(box)

        applied_params = self.apply_params(ff_params)

        # from inspecting source: arg4-arg7 bools mean compute_du_dx, compute_du_dp, compute_du_dl, compute_u
        selection = (False, False, False, True)

        U = 0.0
        for (impl, params) in zip(self.unbound_impls, applied_params):
            U += impl.execute_selective(x, params, box, lam, *selection)[3]
        return U


    def execute_dU_dx(self, x, lam, ff_params, box=None):
        """TODO: reduce code duplication between execute_U, execute_dU_dx, execute_dU_dlam"""
        box = self.handle_optional_box(box)

        applied_params = self.apply_params(ff_params)

        # from inspecting source: arg4-arg7 bools mean compute_du_dx, compute_du_dp, compute_du_dl, compute_u
        selection = (True, False, False, False)

        dU_dx = 0.0
        for (impl, params) in zip(self.unbound_impls, applied_params):
            dU_dx += impl.execute_selective(x, params, box, lam, *selection)[0]
        return dU_dx

    def execute_dU_dlam(self, x, lam, ff_params, box=None):
        """TODO: reduce code duplication between execute_U, execute_dU_dx, execute_dU_dlam"""
        box = self.handle_optional_box(box)

        applied_params = self.apply_params(ff_params)

        # from inspecting source: arg4-arg7 bools mean compute_du_dx, compute_du_dp, compute_du_dl, compute_u
        selection = (False, False, True, False)

        dU_dlam = 0.0
        for (impl, params) in zip(self.unbound_impls, applied_params):
            dU_dlam += impl.execute_selective(x, params, box, lam, *selection)[2]
        return dU_dlam

    def execute_dU_dparams(self, x, lam, ff_params, box=None):
        box = self.handle_optional_box(box)
        applied_params = self.apply_params(ff_params)

        # TODO: Oh, I should re-organize this slightly: can naturally get d U_component / d ff_params
        #   for each U_component in self.unbound_impls

        raise(NotImplementedError)



def construct_differentiable_interface(sys_params, unbound_potentials):
    """Construct a differentiable function u_fxn(x, lam, params) -> float

    Intent: support grad(u_fxn, argnums=(0,1,2))(x, lam, params)
    Status: supports grad(u_fxn, argnums=(0,1))(x, lam, params)

    TODO: defjvp w.r.t. params
    TODO: use .execute_selective(...) instead of .execute(...)[selection_key]
    """

    potential_energy_model = UnboundPotentialEnergyModel(sys_params, unbound_potentials)

    @custom_jvp
    def u_fxn(x, lam, params):
        return potential_energy_model.execute(x, lam, params)['val']

    u_fxn.defjvps(
        lambda x_dot, primal_out, x, lam, params:
        np.sum(x_dot * potential_energy_model.execute(x, lam, params)['du_dx']),
        lambda lam_dot, primal_out, x, lam, params:
        lam_dot * potential_energy_model.execute(x, lam, params)['du_dl'],
        None,
        #   lambda params_dot, primal_out, x, lam, params:
        #       { key: np.sum(params_dot[key] * potential_energy_model.execute(x, lam, params)['du_dparams'][key]) for key in params },
        # TODO: TypeError: Value {'HarmonicAngle': 0.0, 'HarmonicBond': 0.0, 'NonbondedInterpolated': 0.0,
        #   'PeriodicTorsion': 0.0} with type <class 'dict'> is not a valid JAX type
    )

    return u_fxn
