from jax import config

config.update("jax_enable_x64", True)

from jax import custom_jvp
from jax import numpy as np

from timemachine.lib.potentials import SummedPotential


def _make_selection_mask(compute_du_dx=False, compute_du_dp=False, compute_du_dl=False, compute_u=False):
    requested_outputs = []
    if compute_du_dx:
        requested_outputs.append("dudx")
    if compute_du_dp:
        requested_outputs.append("dudp")
    if compute_du_dl:
        requested_outputs.append("dudl")
    if compute_u:
        requested_outputs.append("u")

    print(f'calling execute_selective(..., {", ".join(requested_outputs)})')
    return (compute_du_dx, compute_du_dp, compute_du_dl, compute_u)


def wrap_impl(impl, pack=lambda x: x):
    """Construct a differentiable function U(x, params, box, lam) -> float
    from a single unbound potential
    """

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


def construct_differentiable_interface(unbound_potentials, precision=np.float32):
    """Construct a differentiable function U(x, params, box, lam) -> float
    from a collection of unbound potentials

    >>> U = construct_differentiable_interface(unbound_potentials)
    >>> _ = grad(U, (0,1,3))(coords, sys_params, box, lam)

    This implementation computes the sum of the component potentials in Python
    """
    impls = [ubp.unbound_impl(precision) for ubp in unbound_potentials]
    U_s = [wrap_impl(impl) for impl in impls]

    def U(coords, params, box, lam):
        return np.sum(np.array([U_i(coords, p_i, box, lam) for (U_i, p_i) in zip(U_s, params)]))

    return U


def construct_differentiable_interface_fast(unbound_potentials, params, precision=np.float32):
    """Construct a differentiable function U(x, params, box, lam) -> float
    from a collection of unbound potentials

    >>> U = construct_differentiable_interface(unbound_potentials, params)
    >>> _ = grad(U, (0,1,3))(coords, sys_params, box, lam)

    This implementation computes the sum of the component potentials in C++ using the SummedPotential custom op
    """
    impl = SummedPotential(unbound_potentials, params).unbound_impl(precision)

    def pack(params):
        return np.concatenate([ps.reshape(-1) for ps in params])

    U = wrap_impl(impl, pack)

    return U
