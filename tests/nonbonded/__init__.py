import jax.numpy as jnp
import numpy as np


def gen_params_with_4d_offsets(rng: np.random.Generator, params, w_min, w_max, n):
    for _ in range(n):
        w_coords = rng.uniform(w_min, w_max, (params.shape[0],))
        yield jnp.asarray(params).at[:, 3].set(w_coords)
