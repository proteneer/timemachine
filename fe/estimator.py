from collections import namedtuple

import functools
import jax
import numpy as np

from typing import Tuple, List, Any

import jax.numpy as jnp

FreeEnergyModel = namedtuple(
    "FreeEnergyModel",
    ["dGs", "dG_grads"]
)

gradient = List[Any] # TODO: make this more descriptive of dG_grad structure

def _deltaG(model, sys_params) -> Tuple[Tuple[float, List], np.array]:

    dG_grad = []
    for lhs, rhs in zip(model.dG_grads[0][0], model.dG_grads[-1][-1]):
        dG_grad.append(rhs - lhs)

    return np.sum(model.dGs), dG_grad


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def deltaG(model, sys_params) -> Tuple[float, List]:
    return _deltaG(model=model, sys_params=sys_params)[0]

def deltaG_fwd(model, sys_params) -> Tuple[Tuple[float, List], np.array]:
    """same signature as DeltaG, but returns the full tuple"""
    return _deltaG(model=model, sys_params=sys_params)

def deltaG_bwd(model, residual, grad) -> Tuple[np.array]:
    """Note: nondiff args must appear first here, even though one of them appears last in the original function's signature!
    """
    # residual are the partial dG / partial dparams for each term
    # grad is the adjoint of dG w.r.t. loss: partial L/partial dG
    return ([grad*r for r in residual],)

deltaG.defvjp(deltaG_fwd, deltaG_bwd)