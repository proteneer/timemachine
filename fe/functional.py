from jax import config

config.update("jax_enable_x64", True)

from jax import grad, numpy as np
from jax import custom_jvp

from collections import OrderedDict


def stringify_ubp(ubp):
    return ubp.__class__.__name__


class UnboundPotentialEnergyModel:
    def __init__(self, sys_params, unbound_potentials, precision=np.float64, box=np.eye(3) * 100):
        self.sys_params = sys_params
        self.unbound_potentials = unbound_potentials
        self.ubp_cache = dict()
        self.precision = precision
        self._initialize()
        self.box = box

    def _initialize(self):
        for component_params, unbound_pot in zip(self.sys_params, self.unbound_potentials):
            key = stringify_ubp(unbound_pot)

            if key not in self.ubp_cache:
                impl = unbound_pot.unbound_impl(self.precision)
                self.ubp_cache[key] = impl

    @property
    def all_impls(self):
        """List of impl, e.g. as required by context constructor"""
        return list(self.ubp_cache.values())

    def execute(self, x, lam, params_dict, box=None):
        if box is None:
            box = self.box

        Us, dU_dxs, dU_dls = [], [], []
        dU_dparamses = OrderedDict()
        for key in self.ubp_cache:
            dU_dx, dU_dparams, dU_dl, U = self.ubp_cache[key].execute(x, params_dict[key], box, lam)

            Us.append(U)
            dU_dxs.append(dU_dx)
            dU_dparamses[key] = dU_dparams
            dU_dls.append(dU_dl)

        result = dict(
            val=np.sum(np.array(Us)),
            du_dx=np.sum(np.array(dU_dxs), 0),
            du_dparams=dU_dparamses,  # don't reduce -- will be a dict of arrays of different shape
            du_dl=np.sum(np.array(dU_dls))
        )

        return result


def construct_differentiable_interface(sys_params, unbound_potentials):
    """Construct a differentiable function u_fxn(x, lam, params) -> float

    Intent: support grad(u_fxn, argnums=(0,1,2))(x, lam, params)
    Status: supports grad(u_fxn, argnums=(0,1))(x, lam, params)

    TODO: defjvp w.r.t. params
    TODO: use .execute_selective(...) instead of .execute(...)[selection_key]
    """

    potential_energy_model = UnboundPotentialEnergyModel(sys_params, unbound_potentials)
    params_dict = OrderedDict()
    for ubp, params in zip(unbound_potentials, sys_params):
        params_dict[stringify_ubp(ubp)] = params

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
