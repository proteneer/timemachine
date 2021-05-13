from jax import config

config.update("jax_enable_x64", True)

from jax import grad, numpy as np
from jax import custom_jvp


def _make_selection_mask(compute_du_dx=False, compute_du_dp=False, compute_du_dl=False, compute_u=False):
    return (compute_du_dx, compute_du_dp, compute_du_dl, compute_u)


def wrap_impl(impl):
    @custom_jvp
    def U(coords, params, box, lam):
        selection = _make_selection_mask(compute_u=True)
        result_tuple = impl.execute_selective(coords, params, box, lam, *selection)
        return result_tuple[3]

    def U_jvp_x(coords_dot, primal_out, coords, params, box, lam):
        selection = _make_selection_mask(compute_du_dx=True)
        result_tuple = impl.execute_selective(coords, params, box, lam, *selection)
        return np.sum(coords_dot * result_tuple[0])

    def U_jvp_params(params_dot, primal_out, coords, params, box, lam):
        selection = _make_selection_mask(compute_du_dp=True)
        result_tuple = impl.execute_selective(coords, params, box, lam, *selection)
        return np.sum(params_dot * result_tuple[1])

    def U_jvp_lam(lam_dot, primal_out, coords, params, box, lam):
        selection = _make_selection_mask(compute_du_dl=True)
        result_tuple = impl.execute_selective(coords, params, box, lam, *selection)
        return np.sum(lam_dot * result_tuple[2])

    U.defjvps(U_jvp_x, U_jvp_params, None, U_jvp_lam)

    return U


def construct_differentiable_interface(unbound_potentials, precision=np.float64):
    """Construct a differentiable function U(x, params, box, lam) -> float

    >>> U = construct_differentiable_interface(unbound_potentials)
    >>> _ = grad(U, (0,1,3))(coords, sys_params, box, lam)
    """
    impls = [ubp.unbound_impl(precision) for ubp in unbound_potentials]
    U_s = [wrap_impl(impl) for impl in impls]

    def U(coords, params, box, lam):
        return np.sum(np.array([U_i(coords, p_i, box, lam) for (U_i, p_i) in zip(U_s, params)]))

    return U
