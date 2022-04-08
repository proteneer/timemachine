from jax import config

config.update("jax_enable_x64", True)

from typing import Tuple

from jax import custom_jvp
from jax import numpy as np
from jax.interpreters.partial_eval import Tracer

from timemachine.lib.potentials import SummedPotential


def _make_selection_mask(compute_du_dx=False, compute_du_dp=False, compute_du_dl=False, compute_u=False):
    return (compute_du_dx, compute_du_dp, compute_du_dl, compute_u)


def wrap_impl(impl, pack=lambda x: x):
    """Construct a differentiable function U(coords, params, box, lam) -> float
    from a single unbound potential
    """

    @custom_jvp
    def U(coords, params, box, lam) -> float:
        selection = _make_selection_mask(compute_u=True)
        result_tuple = impl.execute_selective(coords, pack(params), box, lam, *selection)
        return result_tuple[3]

    @U.defjvp
    def U_jvp(primals, tangents) -> Tuple[float, float]:
        # naming convention (x, x_dot), (primal_out, tangent_out) follow the jax custom jvp documentation example
        # https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#custom-jvps-with-jax-custom-jvp

        # unpack inputs
        coords, _params, box, lam = primals
        coords_dot, _params_dot, box_dot, lam_dot = tangents

        # handle case where _params is a list of (maybe traced) arrays
        params, params_dot = pack(_params), pack(_params_dot)

        # inspect tangent types to determine which derivatives are being requested
        def derivative_requested(array_t):
            return isinstance(array_t, Tracer)

        selection = _make_selection_mask(
            compute_du_dx=derivative_requested(coords_dot),
            compute_du_dp=derivative_requested(params_dot),
            compute_du_dl=derivative_requested(lam_dot),
            compute_u=True,
        )

        if derivative_requested(box_dot):
            raise RuntimeError("box derivatives not supported!")

        # call custom op once
        result_tuple = impl.execute_selective(coords, params, box, lam, *selection)

        # unpack result tuple
        primal_out = result_tuple[3]
        coords_grad, params_grad, lam_grad = result_tuple[:3]

        # Jacobian-vector products
        def explicit_jvp(gradient, tangent):
            assert gradient.shape == tangent.shape
            J = gradient.flatten().reshape(1, gradient.size)
            v = tangent.flatten()
            return np.matmul(J, v)

        tangent_out = np.zeros(1)
        if derivative_requested(coords_dot):
            tangent_out += explicit_jvp(coords_grad, coords_dot)
        if derivative_requested(params_dot):
            tangent_out += explicit_jvp(params_grad, params_dot)
        if derivative_requested(lam_dot):
            tangent_out += explicit_jvp(np.array(lam_grad), np.array(lam_dot))

        # specific to case of scalar-valued U: float64[1] -> float64, so primal and tangent have equal shapes/dtypes
        tangent_out = np.sum(tangent_out)

        return primal_out, tangent_out

    return U


def construct_differentiable_interface(unbound_potentials, precision=np.float32):
    """Construct a differentiable function U(coords, params, box, lam) -> float
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
    """Construct a differentiable function U(coords, params, box, lam) -> float
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
