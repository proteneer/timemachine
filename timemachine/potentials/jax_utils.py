import jax.numpy as np


def lambda_to_w(lamb, lamb_flags, cutoff):
    d4_i = np.where(lamb_flags == 1, lamb, 0.0)
    d4_d = np.where(lamb_flags == -1, cutoff + lamb, 0.0)
    d4 = d4_i + d4_d
    return d4

def convert_to_4d(x3, lamb, lamb_flags, cutoff):
    d4 = lambda_to_w(lamb, lamb_flags, cutoff)
    d4 = np.expand_dims(d4, axis=-1)
    x4 = np.concatenate((x3, d4), axis=1)
    return x4

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
    dims = ri.shape[-1]

    # assert box is not None

    # box is None for harmonic bonds, not None for nonbonded terms
    if box is not None:
        for d in range(dims):
            diff -= box[d]*np.floor(np.expand_dims(diff[...,d], axis=-1)/box[d][d]+0.5)

    return diff

def distance(ri, rj, box=None):
    dxdydz = np.power(delta_r(ri, rj, box), 2)
    # np.linalg.norm nans but this doesn't
    dij = np.sqrt(np.sum(dxdydz, axis=-1))
    return dij

