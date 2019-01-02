import numpy as np
import tensorflow as tf
from tensorflow.python.ops.parallel_for.gradients import jacobian

from timemachine.force import ConservativeForce
from timemachine import nn_layers
from timemachine.nn_utils import FLOAT_TYPE, EPSILON


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

    def __init__(self,
        params,
        layer_param_idxs,
        initial_self_ixn_param_idxs, # featurization parameters
        compressed_atom_types,
        rbf_centers,
        rbf_spacing):
        """
        Implements a rotationally and translationally equivariant
        tensorfield neural-network inspired by Nathaniel Thomas and 
        Tess Smidt: https://arxiv.org/abs/1802.08219

        Parameters
        ----------
        params: list of tf.Variables
            used by param_idxs to construct the energy

        self_ixn_param_idxs: [w0, b0] # 1D embedding

        layer_param_idxs: list of layer parameters
            [
                0x1: [
                    0 [convolution 0x0: [w0, w1, b0, b1],
                    convolution 0x1: [w0, w1, b0, b1],
                    convolution 1x0: None,
                    convolution 1x1: None]
                    1 [self-ixn: L0: w0, b0, L1: w0 ]
                    2 [nonlin_bias: L0: b0]#  
                ]
                1x2: [
                    convolution 0x0: [w0, w1, b0, b1]
                    convolution 0x1: [w0, w1, b0, b1]
                    convolution 1x0: [w0, w1, b0, b1]
                    convolution 1x1: [w0, w1, b0, b1]
                    self-ixn: L0: [w0, b0], L1: [w0]
                    nonlin_bias: L0: [b0]
                ]
                ...
                2xf: [
                    convolution 0x0: [w0, w1, b0, b1],
                    convolution 0x1: None,
                    convolution 1x0: [w0, w1, b0, b1],
                    convolution 1x1: None,
                    self-ixn: L0: [w0, b0], L1: None
                    nonlin_bias: L0: [b0]
                ]
            ]

        """
        self.params = params
        self.layer_param_idxs = layer_param_idxs

        self.rbf_spacing = rbf_spacing
        self.centers = rbf_centers

        w_si_idx, b_si_idx = initial_self_ixn_param_idxs
        num_atom_types = np.amax(compressed_atom_types) + 1 #????
        input_one_hot = tf.cast(tf.one_hot(compressed_atom_types, num_atom_types), dtype=tf.float64)
        self.embed = nn_layers.self_interaction_layer_with_biases(
            tf.reshape(input_one_hot, [-1, num_atom_types, 1]),
            self.params[w_si_idx],
            self.params[b_si_idx]
        )

    def mixed_partials(self, conf):
        # optimized version to speed things up a little bit.
        grads = self.gradients(conf)
        reverse_shaped = jacobian(grads, self.params, use_pfor=False) 
        properly_shaped = []
        for p in reverse_shaped:
            if len(p.get_shape()) == 3:
                properly_shaped.append(tf.transpose(p, perm=(2,0,1)))
            if len(p.get_shape()) == 4:
                # properly_shaped.append(tf.reshape(fixed, [-1, fixed.shape[2], fixed.shape[3]]))
                properly_shaped.append(tf.transpose(p, perm=(2,3,0,1)))
        return properly_shaped

    def energy(self, conf):
        ri = tf.expand_dims(conf, axis=1)
        rj = tf.expand_dims(conf, axis=0)
        rij = ri - rj
      
        unit_vectors = rij / tf.expand_dims(tf.norm(rij, axis=-1) + EPSILON, axis=-1) # can we get rid of this EPSILON?

        dij = norm_with_epsilon(rij, axis=-1)

        gamma = 1. / self.rbf_spacing
        rbf = tf.exp(-gamma * tf.square(tf.expand_dims(dij, axis=-1) - self.centers))

        embed = self.embed
        input_tensor_list = {0: [embed]}

        num_layers = len(self.layer_param_idxs)
        for layer in range(num_layers):

            c0x0_idxs, c0x1_idxs, c1x0_idxs, c1x1_idxs = self.layer_param_idxs[layer][0]
            si_w0_idx, si_b0_idx, si_w1_idx = self.layer_param_idxs[layer][1]
            nonlin_bias_idx = self.layer_param_idxs[layer][2]

            # (ytz): all the Nones are to deal with first and last layer not having
            # connections to avoid None in dEdp calculations. So it's a little messy.
            def convert(p_list):
                if p_list is not None:
                    return [self.params[idx] for idx in p_list]
                else:
                    return None

            def single_convert(params, idx):
                if idx is not None:
                    return params[idx]
                else:
                    return None

            input_tensor_list0 = nn_layers.convolution(
                input_tensor_list,
                rbf,
                rij,
                convert(c0x0_idxs),
                convert(c0x1_idxs),
                convert(c1x0_idxs),
                convert(c1x1_idxs),
            )

            input_tensor_list1 = nn_layers.concatenation(input_tensor_list0)
            input_tensor_list2 = nn_layers.self_interaction(
                input_tensor_list1,
                self.params[si_w0_idx],
                self.params[si_b0_idx],
                single_convert(self.params, si_w1_idx),
            )
            input_tensor_list3 = nn_layers.nonlinearity(
                input_tensor_list2,
                single_convert(self.params, nonlin_bias_idx),
                nonlin=tf.nn.elu
            )
            input_tensor_list = input_tensor_list3

        # (ytz): useful for debugging, leave this here for now.
        # tot_params = 0
        # for v in tf.trainable_variables():
        #     if len(v.shape) == 2:
        #         tot_params += v.shape[0]*v.shape[1]
        #     elif len(v.shape) == 1:
        #         tot_params += v.shape[0]
        #     else:
        #         raise Exception("fail")
        #     print(v.name, v.shape)
        # print("total params in tfn", tot_params)


        atomic_energies = input_tensor_list[0][0]
        return tf.reduce_sum(atomic_energies)