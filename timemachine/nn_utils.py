import tensorflow as tf
import numpy as np
import scipy.linalg

FLOAT_TYPE = tf.float64
NP_FLOAT_TYPE = np.float64
EPSILON = 1e-8


def get_eijk():
    """
    Constant Levi-Civita tensor

    Returns:
        tf.Tensor of shape [3, 3, 3]
    """
    eijk_ = np.zeros((3, 3, 3))
    eijk_[0, 1, 2] = eijk_[1, 2, 0] = eijk_[2, 0, 1] = 1.
    eijk_[0, 2, 1] = eijk_[2, 1, 0] = eijk_[1, 0, 2] = -1.
    return tf.constant(eijk_, dtype=FLOAT_TYPE)


def norm_with_epsilon(input_tensor, axis=None, keep_dims=False):
    """
    Regularized norm

    Args:
        input_tensor: tf.Tensor

    Returns:
        tf.Tensor normed over axis
    """
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(input_tensor), axis=axis, keep_dims=keep_dims), EPSILON))


def ssp(x):
    """
    Shifted soft plus nonlinearity.

    Args:
        x: tf.Tensor

    Returns:
        tf.Tensor of same shape as x 
   """
    return tf.log(0.5 * tf.exp(x) + 0.5)


def rotation_equivariant_nonlinearity(x, biases=None, nonlin=ssp):
    """
    Rotation equivariant nonlinearity.

    The -1 axis is assumed to be M index (of which there are 2 L + 1 for given L).

    Args:
        x: tf.Tensor with channels as -2 axis and M as -1 axis.

    Returns:
        tf.Tensor of same shape as x with 3d rotation-equivariant nonlinearity applied.
    """
    shape = x.get_shape().as_list()
    representation_index = shape[-1]
    if representation_index == 1:
        assert biases is None
        return nonlin(x)
    else:
        assert biases is not None
        norm = norm_with_epsilon(x, axis=-1)
        nonlin_out = nonlin(tf.nn.bias_add(norm, biases))
        factor = tf.divide(nonlin_out, norm)
        # Expand dims for representation index.
        return tf.multiply(x, tf.expand_dims(factor, axis=-1))
    


def difference_matrix(geometry):
    """
    Get relative vector matrix for array of shape [N, 3].

    Args:
        geometry: tf.Tensor with Cartesian coordinates and shape [N, 3]

    Returns:
        Relative vector matrix with shape [N, N, 3]
    """
    # [N, 1, 3]
    ri = tf.expand_dims(geometry, axis=1)
    # [1, N, 3]
    rj = tf.expand_dims(geometry, axis=0)
    # [N, N, 3]
    rij = ri - rj
    return rij


def distance_matrix(geometry):
    """
    Get relative distance matrix for array of shape [N, 3].

    Args:
        geometry: tf.Tensor with Cartesian coordinates and shape [N, 3]

    Returns:
        Relative distance matrix with shape [N, N]
    """
    # [N, N, 3]
    rij = difference_matrix(geometry)
    # [N, N]
    dij = norm_with_epsilon(rij, axis=-1)
    return dij


