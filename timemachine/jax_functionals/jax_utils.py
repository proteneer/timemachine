import jax.numpy as np


def rescale_coordinates(
    conf,
    indices,
    box,
    scales):
    
    mol_sizes = np.expand_dims(onp.bincount(indices), axis=1)
    mol_centers = jax.ops.segment_sum(coords, indices)/mol_sizes

    new_centers = mol_centers - box[2]*np.floor(np.expand_dims(mol_centers[...,2], axis=-1)/box[2][2])
    new_centers -= box[1]*np.floor(np.expand_dims(new_centers[...,1], axis=-1)/box[1][1])
    new_centers -= box[0]*np.floor(np.expand_dims(new_centers[...,0], axis=-1)/box[0][0])

    offset = new_centers - mol_centers

    return conf + offset[indices]


def delta_r(ri, rj, box=None):
    diff = ri - rj # this can be either N,N,3 or B,3
    if box is not None:
        diff -= box[2]*np.floor(np.expand_dims(diff[...,2], axis=-1)/box[2][2]+0.5)
        diff -= box[1]*np.floor(np.expand_dims(diff[...,1], axis=-1)/box[1][1]+0.5)
        diff -= box[0]*np.floor(np.expand_dims(diff[...,0], axis=-1)/box[0][0]+0.5)
    return diff

def distance(ri, rj, box=None):
    dxdydz = np.power(delta_r(ri, rj, box), 2)
    # np.linalg.norm nans but this doesn't
    dij = np.sqrt(np.sum(dxdydz, axis=-1))
    return dij

