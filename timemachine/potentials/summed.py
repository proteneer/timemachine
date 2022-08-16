from typing import Any, Callable, Sequence, Tuple

import numpy as np

Array = Any
Conf = Array
Params = Array
Box = Array
Lambda = float
PotentialFxn = Callable[[Conf, Params, Box, Lambda], float]


def summed_potential(
    conf: Array,
    params: Array,
    box: Array,
    lamb: float,
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

    U_fns: list of functions with signature (conf, params, box, lambda) -> energy
        potential terms

    shapes: list of tuple
        shapes of the parameter array input for each potential term (must be same length as U_fns)
    """
    assert len(U_fns) == len(shapes)
    sizes = np.prod(shapes, axis=1)
    paramss = [ps.reshape(shape) for ps, shape in zip(np.split(params, sizes[:-1]), shapes)]
    return sum(U_fn(conf, ps, box, lamb) for U_fn, ps in zip(U_fns, paramss))
