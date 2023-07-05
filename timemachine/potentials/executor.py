from functools import partial
from typing import Any, List, Optional, Tuple, cast

import jax
import jax.numpy as jnp
from jax.core import Tracer
from numpy.typing import NDArray

from timemachine.lib import custom_ops


class PotentialExecutor:
    """Interface to execute a set of potentials efficiently for host side code with the option of running the potentials in parallel"""

    def __init__(self, N: int, parallel: bool = True):
        self._N = N
        self._runner = custom_ops.PotentialExecutor(N, parallel)

    def execute_bound(
        self,
        bps: List[custom_ops.BoundPotential],
        coords: NDArray,
        box: NDArray,
        compute_du_dx: bool = True,
        compute_u: bool = True,
    ) -> Tuple[Optional[NDArray], Optional[float]]:
        """Execute a list of bound potentials on a set of coords and box"""
        assert coords.shape[0] == self._N, "Number of coordinates don't match"
        du_dx, u = self._runner.execute_bound(bps, coords, box, compute_du_dx, compute_u)
        return du_dx, u

    def execute(self, pots: List[custom_ops.Potential], coords: NDArray, params: List[NDArray], box: NDArray) -> float:
        """Execute a list of potentials on a set of coords and box.

        Supports jax.grad to compute the du_dx and du_dp components.
        """
        assert coords.shape[0] == self._N, "Number of coordinates don't match"
        if len(pots) != len(params):
            raise RuntimeError("Number of potentials and params must agree")
        param_sizes = [len(p) for p in params]

        flattened_params = jnp.concatenate([p.reshape(-1) for p in params])
        u = _call_runner_unbound(self._runner, pots, param_sizes, coords, flattened_params, box)
        return cast(float, u)


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def _call_runner_unbound(
    runner: custom_ops.PotentialExecutor,
    pots: List[custom_ops.Potential],
    param_sizes,
    coords: NDArray,
    params: NDArray,
    box: NDArray,
) -> float:
    _, _, u = runner.execute_unbound(pots, param_sizes, coords, params, box, False, False, True)
    return u


# Add JAX custom derivative rules
# https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#custom-jvps-with-jax-custom-jvp
@_call_runner_unbound.defjvp
def _(
    runner: custom_ops.PotentialExecutor, pots: List[custom_ops.Potential], param_sizes, primals, tangents
) -> Tuple[Any, Any]:
    x, params, box = primals
    dx, dp, dbox = tangents

    if isinstance(dbox, Tracer):
        raise RuntimeError("box derivatives not supported")

    compute_du_dx = isinstance(dx, Tracer)
    compute_du_dp = isinstance(dp, Tracer)

    du_dx, du_dp, u = runner.execute_unbound(pots, param_sizes, x, params, box, compute_du_dx, compute_du_dp, True)

    tangent_out = jnp.zeros_like(u)

    if compute_du_dx:
        tangent_out += jnp.sum(du_dx * dx)
    if compute_du_dp:
        tangent_out += jnp.sum(du_dp * dp)

    return u, tangent_out
