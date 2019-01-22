# implements various derivatives of various use
import tensorflow as tf
from tensorflow.python.ops.parallel_for.gradients import jacobian

def densify(op):
    if isinstance(op, tf.IndexedSlices):
        return tf.unsorted_segment_sum(
            op.values,
            op.indices,
            op.dense_shape[0])
    else:
        return op



def list_jacobian(outputs, inputs):
    """
    Parameters
    ----------
    outputs: tf.Tensor
    inputs: list of tf.Tensor

    Returns
    -------
    list of jacobians [
        tf.Tensor(p0_d0,p0_d1,...,N,3),
        tf.Tensor(p1_d0,p1_d2,...,N,3),
        ...
    ]

    """
    # This is a slightly more advanced version of tensorflow's jacobian system that allows
    # for sparse gradients as well as automatically reshaping the results if outputs is a list.

    # taken from tf src gradients_impl.py _IndexedSlicesToTensor 
    densify(outputs)

    output_dims = list(range(len(outputs.get_shape().as_list()))) # [0,1]
    n_out_dims = len(output_dims)

    # if isinstance(reverse_shaped, tf.Tensor):
        # shove to a list
        # reverse_shaped = [reverse_shaped]

    result = []
    for inp, jac in zip(inputs, jacobian(outputs, inputs, use_pfor=False)):

        input_dims = list(range(len(inp.get_shape().as_list()))) # [0,1]
        perm = [(idx + n_out_dims) for idx in input_dims] + output_dims # generate permutation indices
        result.append(tf.transpose(jac, perm=perm))

    return result

def compute_ghm(energy_op, x, params):
    """
    Computes gradients, hessians, mixed_partials in one go
    """
    grads = tf.gradients(energy_op, x)[0]
    hess = tf.hessians(energy_op, x)[0]
    mp = list_jacobian(grads, params)
    return grads, hess, mp