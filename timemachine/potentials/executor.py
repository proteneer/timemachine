from functools import partial
from typing import Any, List, Optional, Tuple, cast

import jax
import jax.numpy as jnp
from jax.core import Tracer
from numpy.typing import NDArray

from timemachine.lib import custom_ops


class PotentialExecutor:
    """Interface to execute a set of potentials efficiently from python with the option of running the potentials in parallel"""

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
        """Execute a list of bound potentials on a set of coords and box

        Parameters
        ----------

        bps: list of custom_ops.BoundPotentials
            Potentials setup to run on the system
        coords: ndarray
            Coordinates of the system to evaluate
        box: ndarray
            Box of the system to evaluate
        compute_du_dx: bool
            Whether or not to compute du_dx, returns None in place of du_dx if False
        compute_u: bool
            Whether or not to compute the energy, returns None in place of u if False

        Returns
        -------
        Optional ndarray
            The du_dx of the system if compute_du_dx is True

        Optional float
            Eergy of the system if compute_u is True
        """
        assert len(bps) > 0, "Must provide at least one potential"
        assert coords.shape[0] == self._N, "Number of coordinates don't match"
        du_dx, u = self._runner.execute_bound(bps, coords, box, compute_du_dx, compute_u)
        return du_dx, u

    def execute(self, pots: List[custom_ops.Potential], coords: NDArray, params: List[NDArray], box: NDArray) -> float:
        """Execute a list of potentials on a set of coords, parameters and box.

        Supports jax.grad to compute the du_dx and du_dp components.

        Parameters
        ----------

        pots: list of custom_ops.Potentials
            Potentials setup to run on the system
        coords: ndarray
            Coordinates of the system to evaluate
        params: list of np.ndarray
            The params of each potential
        box: ndarray
            Box of the system to evaluate

        Returns
        -------
        float
            Energy of the system
        """
        assert len(pots) > 0, "Must provide at least one potential"
        assert coords.shape[0] == self._N, "Number of coordinates don't match"
        assert len(pots) == len(params), "Number of potentials and params don't match"

        param_sizes = jnp.array([p.size for p in params], dtype=jnp.int32)

        flattened_params = jnp.concatenate([p.reshape(-1) for p in params])
        assert jnp.sum(param_sizes) == flattened_params.size

        u = _call_runner_unbound(self._runner, pots, param_sizes, coords, flattened_params, box)
        return u


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def _call_runner_unbound(
    runner: custom_ops.PotentialExecutor,
    pots: List[custom_ops.Potential],
    param_sizes,
    coords: NDArray,
    params: NDArray,
    box: NDArray,
) -> float:
    _, _, u = runner.execute_unbound(
        pots, param_sizes, coords, params, box, compute_du_dx=False, compute_du_dp=False, compute_u=True
    )
    return cast(float, u)


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

    du_dx, du_dp, u = runner.execute_unbound(
        pots, param_sizes, x, params, box, compute_du_dx=compute_du_dx, compute_du_dp=compute_du_dp, compute_u=True
    )

    tangent_out = jnp.zeros_like(u)

    if compute_du_dx:
        tangent_out += jnp.sum(du_dx * dx)
    if compute_du_dp:
        tangent_out += jnp.sum(du_dp * dp)

    return u, tangent_out
