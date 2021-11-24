from jax import config

config.update("jax_enable_x64", True)

from jax import custom_jvp, numpy as np

from timemachine.lib import custom_ops


def _make_selection_mask(compute_du_dx=False, compute_du_dp=False, compute_du_dl=False, compute_u=False):
    return (compute_du_dx, compute_du_dp, compute_du_dl, compute_u)


def construct_differentiable_interface(unbound_potentials, params, precision=np.float32):
    """Construct a differentiable function U(x, params, box, lam) -> float

    >>> U = construct_differentiable_interface(unbound_potentials)
    >>> _ = grad(U, (0,1,3))(coords, sys_params, box, lam)
    """
    impls = [ubp.unbound_impl(precision) for ubp in unbound_potentials]
    sizes = [ps.size for ps in params]
    impl = custom_ops.SummedPotential(impls, sizes)

    def pack(params):
        return np.concatenate([ps.reshape(-1) for ps in params])

    @custom_jvp
    def U(coords, params, box, lam):
        selection = _make_selection_mask(compute_u=True)
        result_tuple = impl.execute_selective(coords, pack(params), box, lam, *selection)
        return result_tuple[3]

    def U_jvp_x(coords_dot, _, coords, params, box, lam):
        selection = _make_selection_mask(compute_du_dx=True)
        result_tuple = impl.execute_selective(coords, pack(params), box, lam, *selection)
        return np.sum(coords_dot * result_tuple[0])

    def U_jvp_params(params_dot, _, coords, params, box, lam):
        selection = _make_selection_mask(compute_du_dp=True)
        result_tuple = impl.execute_selective(coords, pack(params), box, lam, *selection)
        return np.sum(pack(params_dot) * result_tuple[1])

    def U_jvp_lam(lam_dot, _, coords, params, box, lam):
        selection = _make_selection_mask(compute_du_dl=True)
        result_tuple = impl.execute_selective(coords, pack(params), box, lam, *selection)
        return np.sum(lam_dot * result_tuple[2])

    U.defjvps(U_jvp_x, U_jvp_params, None, U_jvp_lam)

    return U
