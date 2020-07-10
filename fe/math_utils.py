import jax.numpy as jnp

def trapz(y, x, dx=1.0, axis=-1):
    """
    Isomorphic API to numpy's trapz. Refer to np.trapz for documentation.

    y and x must be numpy arrays.
    """
    d = jnp.diff(x)
    # reshape to correct shape
    shape = [1]*y.ndim
    shape[axis] = d.shape[0]
    d = d.reshape(shape)

    nd = y.ndim
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)

    ret = (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)
    return ret
