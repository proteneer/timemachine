import jax.numpy as np


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

