from typing import Callable, Optional, TypeAlias

import jax
import numpy as np

Array: TypeAlias = jax.Array | np.ndarray
Conf: TypeAlias = Array
Params: TypeAlias = Array
Box: TypeAlias = Array

PotentialFxn = Callable[[Conf, Params, Optional[Box]], float | jax.Array]
BoundPotentialFxn = Callable[[Conf, Optional[Box]], float | jax.Array]
