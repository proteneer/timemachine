from typing import List, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from .types import Box, Conf, Params, PotentialFxn


def summed_potential(
    conf: Conf,
    params: Params,
    box: Optional[Box],
    U_fns: Sequence[PotentialFxn],
    shapes: Sequence[Tuple],
):
    """Reference implementation of the custom_ops SummedPotential.

    Parameters
    ----------
    conf: array (N, 3)
        conformation

    params: array (P,)
        flattened array of parameters for all potential terms

    box: array (3, 3)
        periodic box

    U_fns: list of functions with signature (conf, params, box) -> energy
        potential terms

    shapes: list of tuple
        shapes of the parameter array input for each potential term (must be same length as U_fns)
    """
    assert len(U_fns) == len(shapes)
    unflattened_params = unflatten_params(params, shapes)
    return jnp.sum(jnp.array([U_fn(conf, ps, box) for U_fn, ps in zip(U_fns, unflattened_params)]))


def unflatten_params(params: Params, shapes: Sequence[Tuple[int, ...]]) -> List[Params]:
    sizes = [int(np.prod(shape)) for shape in shapes]
    assert params.shape == (sum(sizes),)
    split_indices = np.cumsum(sizes)
    return [ps.reshape(shape) for ps, shape in zip(np.split(params, split_indices[:-1]), shapes)]


def fanout_summed_potential(
    conf: Conf,
    params: Params,
    box: Optional[Box],
    U_fns: Sequence[PotentialFxn],
):
    """Reference implementation of the custom_ops FanoutSummedPotential.

    Parameters
    ----------
    conf: array (N, 3)
        conformation

    params: array (P,)
        flattened array of parameters shared by each potential term

    box: array (3, 3)
        periodic box

    U_fns: list of functions with signature (conf, params, box) -> energy
        potential terms
    """
    return jnp.sum(jnp.array([U_fn(conf, ps, box) for U_fn, ps in zip(U_fns, jnp.array(params))]))
