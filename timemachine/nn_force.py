import numpy as np
import tensorflow as tf
from timemachine.force import ConservativeForce
from timemachine import nn_layers

from nn_utils import FLOAT_TYPE, EPSILON

def norm_with_epsilon(input_tensor, axis=None, keep_dims=False):
    """
    Regularized norm

    Args:
        input_tensor: tf.Tensor

    Returns:
        tf.Tensor normed over axis
    """
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(input_tensor), axis=axis, keep_dims=keep_dims), EPSILON))


class TensorfieldForce(ConservativeForce):

    def __init__(self, compressed_atom_types):
        """
        Implements a rotationally and translationally equivariant
        tensorfield neural-network inspired by Nathaniel Thomas and 
        Tess Smidt: https://arxiv.org/abs/1802.08219
        """
        self.params = None # parameters are from layers

        rbf_low = 0.
        rbf_high = 4.6
        rbf_count = 4
        self.rbf_spacing = (rbf_high - rbf_low) / rbf_count
        self.centers = tf.cast(tf.lin_space(rbf_low, rbf_high, rbf_count), tf.float64)

        self.compressed_atom_types = compressed_atom_types


    def energy(self, conf):
        ri = tf.expand_dims(conf, axis=1)
        rj = tf.expand_dims(conf, axis=0)
        rij = ri - rj

        num_atom_types = np.amax(self.compressed_atom_types)+1 #????
        input_one_hot = tf.cast(tf.one_hot(self.compressed_atom_types, num_atom_types), dtype=tf.float64)
        unit_vectors = rij / tf.expand_dims(tf.norm(rij, axis=-1) + EPSILON, axis=-1) # can we get rid of this EPSILON?

        dij = norm_with_epsilon(rij, axis=-1)

        gamma = 1. / self.rbf_spacing
        rbf = tf.exp(-gamma * tf.square(tf.expand_dims(dij, axis=-1) - self.centers))
        layer_dims = [5, 5, 5, 1]

        with tf.variable_scope('embed', values=[input_one_hot],  reuse=tf.AUTO_REUSE):
            embed = nn_layers.self_interaction_layer_with_biases(
                tf.reshape(input_one_hot, [-1, num_atom_types, 1]), layer_dims[0]
            )
            input_tensor_list = {0: [embed]}

        num_layers = len(layer_dims) - 1
        for layer in range(num_layers):
            layer_dim = layer_dims[layer + 1]
            with tf.variable_scope('layer' + str(layer), values=[input_tensor_list],  reuse=tf.AUTO_REUSE):
                input_tensor_list0 = nn_layers.convolution(input_tensor_list, rbf, rij)
                input_tensor_list1 = nn_layers.concatenation(input_tensor_list0)
                input_tensor_list2 = nn_layers.self_interaction(input_tensor_list1, layer_dim)
                # if layer == num_layers - 1:
                input_tensor_list3 = nn_layers.nonlinearity(input_tensor_list2, nonlin=tf.nn.relu)
                # else:
                    # input_tensor_list3 = nn_layers.nonlinearity(input_tensor_list2)
                input_tensor_list = input_tensor_list3


        atomic_energies = input_tensor_list[0][0]
        return tf.reduce_sum(atomic_energies), input_tensor_list