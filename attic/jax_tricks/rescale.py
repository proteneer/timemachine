import jax
from jax import numpy as np
import numpy as onp


def rescale_coordinates(conf, indices, box, scales):
    """Note: scales unused"""

    mol_sizes = np.expand_dims(onp.bincount(indices), axis=1)
    mol_centers = jax.ops.segment_sum(conf, indices) / mol_sizes

    new_centers = mol_centers - box[2] * np.floor(np.expand_dims(mol_centers[..., 2], axis=-1) / box[2][2])
    new_centers -= box[1] * np.floor(np.expand_dims(new_centers[..., 1], axis=-1) / box[1][1])
    new_centers -= box[0] * np.floor(np.expand_dims(new_centers[..., 0], axis=-1) / box[0][0])

    offset = new_centers - mol_centers

    return conf + offset[indices]
