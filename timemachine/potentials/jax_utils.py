import jax.numpy as np

def convert_to_4d(x3, lamb, lambda_plane_idxs, lambda_offset_idxs, cutoff):

    # (ytz): this initializes the 4th dimension to a fixed plane adjust by an offset
    # followed by a scaling by cutoff.

    # lambda_plane_idxs are typically 0 or 1 and allows us to turn off an interaction
    # independent of the lambda value.

    # lambda_offset_idxs are typically 0 and 1, and allows us to adjust the w coordinate
    # in a lambda-dependent way.
    d4 = cutoff*(lambda_plane_idxs + lambda_offset_idxs*lamb)
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

    # box is None for harmonic bonds, not None for nonbonded terms
    if box is not None:
        for d in range(dims):
            diff -= box[d]*np.floor(np.expand_dims(diff[...,d], axis=-1)/box[d][d]+0.5)

    return diff

# def distance(ri, rj, box=None, gij=None):
#     # assert box is None
#     if gij is not None:
#         deltas_4d = np.power(ri - rj, 2)   
#         # print(deltas_4d.shape)
#         deltas_3d = deltas_4d[..., :3]
#         # print(deltas_3d.shape)
#         dij_4d = np.sqrt(np.sum(deltas_4d, axis=-1))
#         dij_3d = np.sqrt(np.sum(deltas_3d, axis=-1))

#         # print("shapes", gij.shape, dij_3d.shape, dij_4d.shape)
#         dij = np.where(gij, dij_3d, dij_4d)
#     else:
#         deltas = np.power(ri - rj, 2)
#         dij = np.sqrt(np.sum(deltas, axis=-1))

#     # print(dij)

#     return dij

def distance(ri, rj, box=None):
    dxdydz = np.power(delta_r(ri, rj, box), 2)
    # np.linalg.norm nans but this doesn't
    dij = np.sqrt(np.sum(dxdydz, axis=-1))
    return dij

