from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax.core import Tracer
from numpy.typing import NDArray

from timemachine.lib import custom_ops


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def call_unbound_impl(impl: custom_ops.Potential, conf: NDArray, params: NDArray, box: NDArray) -> float:
    _, _, u = impl.execute(conf, params, box, False, False, True)
    return u


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def call_bound_impl(impl: custom_ops.BoundPotential, conf: NDArray, box: NDArray) -> float:
    _, u = impl.execute(conf, box, compute_du_dx=False)
    return u


# Add JAX custom derivative rules
# https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#custom-jvps-with-jax-custom-jvp
@call_unbound_impl.defjvp
def _(impl: custom_ops.Potential, primals, tangents) -> tuple[Any, Any]:
    x, p, box = primals
    dx, dp, dbox = tangents

    if isinstance(dbox, Tracer):
        raise RuntimeError("box derivatives not supported")

    compute_du_dx = isinstance(dx, Tracer)
    compute_du_dp = isinstance(dp, Tracer)

    du_dx, du_dp, u = impl.execute(x, p, box, compute_du_dx, compute_du_dp, True)

    tangent_out = jnp.zeros_like(u)

    if compute_du_dx:
        tangent_out += jnp.sum(du_dx * dx)
    if compute_du_dp:
        tangent_out += jnp.sum(du_dp * dp)

    return u, tangent_out


@call_bound_impl.defjvp
def _(impl: custom_ops.BoundPotential, primals, tangents) -> tuple[Any, Any]:
    x, box = primals
    dx, dbox = tangents

    compute_du_dx = isinstance(dx, Tracer)

    if isinstance(dbox, Tracer):
        raise RuntimeError("box derivatives not supported")

    du_dx, u = impl.execute(x, box, compute_du_dx, True)

    tangent_out = jnp.zeros_like(u)

    if compute_du_dx:
        tangent_out += jnp.sum(du_dx * dx)

    return u, tangent_out
