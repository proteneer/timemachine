from math import sqrt
import tensorflow as tf
import numpy as np
from timemachine import nn_utils
from timemachine.nn_utils import FLOAT_TYPE, EPSILON

# Layers for 3D rotation-equivariant network.


trainable_vars = {}

def R(inputs,
    conv_params,
    nonlin=tf.nn.relu):
    w1, b1, w2, b2 = conv_params
    with tf.variable_scope(None, "radial_function", values=[inputs]):

        hidden_layer = nonlin(b1 + tf.tensordot(inputs, w1, [[2], [1]]))
        radial = b2 + tf.tensordot(hidden_layer, w2, [[2], [1]])

        # [N, N, output_dim]
        return radial


def unit_vectors(v, axis=-1):
    return v / nn_utils.norm_with_epsilon(v, axis=axis, keep_dims=True)


def Y_2(rij):
    # rij : [N, N, 3]
    # x, y, z : [N, N]
    x = rij[:, :, 0]
    y = rij[:, :, 1]
    z = rij[:, :, 2]
    r2 = tf.maximum(tf.reduce_sum(tf.square(rij), axis=-1), EPSILON)
    # return : [N, N, 5]
    output = tf.stack([x * y / r2,
                       y * z / r2,
                       (-tf.square(x) - tf.square(y) + 2. * tf.square(z)) / (2 * sqrt(3) * r2),
                       z * x / r2,
                       (tf.square(x) - tf.square(y)) / (2. * r2)],
                      axis=-1)
    return output


def F_0(inputs, conv_params, nonlin=tf.nn.relu):
    # [N, N, output_dim, 1]
    # with tf.variable_scope(None, "F_0", values=[inputs]):
    return tf.expand_dims(
        R(inputs, conv_params, nonlin=nonlin),
        axis=-1)


def F_1(inputs, rij, conv_params, nonlin=tf.nn.relu):
    # [N, N, output_dim]
    radial = R(inputs, conv_params, nonlin=nonlin)
    # Mask out for dij = 0
    dij = tf.norm(rij, axis=-1)
    # output_dim = conv_params[0].shape[0]
    # for p in conv_params:
    output_dim = conv_params[2].shape[0]
    condition = tf.tile(tf.expand_dims(dij < EPSILON, axis=-1), [1, 1, output_dim])
    masked_radial = tf.where(condition, tf.zeros_like(radial), radial)
    # [N, N, output_dim, 3]
    # assert 0
    return tf.expand_dims(unit_vectors(rij), axis=-2) * tf.expand_dims(masked_radial, axis=-1)


def F_2(inputs, rij, conv_params, nonlin=tf.nn.relu):
    # [N, N, output_dim]
    radial = R(inputs, conv_params)
    # Mask out for dij = 0
    dij = tf.norm(rij, axis=-1)
    output_dim = conv_params[2].shape[0]
    condition = tf.tile(tf.expand_dims(dij < EPSILON, axis=-1), [1, 1, output_dim])
    masked_radial = tf.where(condition, tf.zeros_like(radial), radial)
    # [N, N, output_dim, 5]
    return tf.expand_dims(Y_2(rij), axis=-2) * tf.expand_dims(masked_radial, axis=-1)


def filter_0(layer_input,
             rbf_inputs,
             conv_params,
             nonlin=tf.nn.relu):
    with tf.variable_scope(None, "F0_to_L", values=[layer_input]):
        # [N, N, output_dim, 1]
        F_0_out = F_0(rbf_inputs, conv_params, nonlin=nonlin)
        # [N, output_dim]
        input_dim = layer_input.get_shape().as_list()[-1]
        # Expand filter axis "j"
        cg = tf.expand_dims(tf.eye(input_dim, dtype=FLOAT_TYPE), axis=-2)
        res = tf.einsum('ijk,abfj,bfk->afi', cg, F_0_out, layer_input)
        return res

# F x I -> O
# 0 x 0 -> 0 scalar product
# 1 x 0 -> 0 not allowed
# 1 x 1 -> 0 dot product
# 1 x 0 -> 1 vector scalar
# 1 x 1 -> 1 cross product

def filter_1_output_0(layer_input,
                      rbf_inputs,
                      rij,
                      conv_params,
                      nonlin=tf.nn.relu):
    with tf.variable_scope(None, "F1_to_0", values=[layer_input]):
        # [N, N, output_dim, 3]
        F_1_out = F_1(rbf_inputs, rij, conv_params, nonlin=nonlin)
        # [N, output_dim, 3]
        if layer_input.get_shape().as_list()[-1] == 1:
            raise ValueError("0 x 1 cannot yield 0")
        elif layer_input.get_shape().as_list()[-1] == 3:
            # 1 x 1 -> 0
            cg = tf.expand_dims(tf.eye(3, dtype=FLOAT_TYPE), axis=0) # dot product
            return tf.einsum('ijk,abfj,bfk->afi', cg, F_1_out, layer_input)
        else:
            raise NotImplementedError("Other Ls not implemented")




def filter_1_output_1(layer_input,
                      rbf_inputs,
                      rij,
                      conv_params,
                      nonlin=tf.nn.relu):
    with tf.variable_scope(None, "F1_to_1", values=[layer_input]):
        # [N, N, output_dim, 3]
        F_1_out = F_1(rbf_inputs, rij, conv_params, nonlin=nonlin)

        if layer_input.get_shape().as_list()[-1] == 1:
            # 0 x 1 -> 1
            cg = tf.expand_dims(tf.eye(3, dtype=FLOAT_TYPE), axis=-1)
            return tf.einsum('ijk,abfj,bfk->afi', cg, F_1_out, layer_input)
        elif layer_input.get_shape().as_list()[-1] == 3:
            # 1 x 1 -> 1
            return tf.einsum('ijk,abfj,bfk->afi', nn_utils.get_eijk(), F_1_out, layer_input)
        else:
            raise NotImplementedError("Other Ls not implemented")


def filter_2_output_2(layer_input,
                      rbf_inputs,
                      rij,
                      conv_params):
    with tf.variable_scope(None, "F2_to_2", values=[layer_input]):
        # [N, N, output_dim, 3]
        F_2_out = F_2(rbf_inputs, rij, conv_params, nonlin=nonlin)
        # [N, output_dim, 5]
        if layer_input.get_shape().as_list()[-1] == 1:
            # 0 x 2 -> 2
            cg = tf.expand_dims(tf.eye(5), axis=-1)
            return tf.einsum('ijk,abfj,bfk->afi', cg, F_2_out, layer_input)
        else:
            raise NotImplementedError("Other Ls not implemented")


def self_interaction_layer_without_biases(inputs, w_si):
    return tf.transpose(tf.einsum('afi,gf->aig', inputs, w_si), perm=[0, 2, 1])

# only used for the initial convolution
def self_interaction_layer_with_biases(inputs, w_si, b_si):
    return tf.transpose(tf.einsum('afi,gf->aig', inputs, w_si) + b_si, perm=[0, 2, 1])


def convolution(
    input_tensor_list,
    rbf,
    unit_vectors,
    c0x0_params,
    c0x1_params,
    c1x0_params,
    c1x1_params):

    assert c0x0_params is not None
    assert len(c0x0_params) == 4

    output_tensor_list = {0: [], 1: []}
    for key in input_tensor_list:
        for i, tensor in enumerate(input_tensor_list[key]):
            output_dim = tensor.get_shape().as_list()[-2]
            tensor = tf.identity(tensor, name="in_tensor")

            assert c0x0_params is not None

            tensor_out = filter_0(
                tensor,
                rbf,
                c0x0_params)
            # print(key, "x 0 -> L tensor_out shapes, T, RBF, O, D", tensor.shape, rbf.shape, tensor_out.shape, output_dim)
            m = 0 if tensor_out.get_shape().as_list()[-1] == 1 else 1
            tensor_out = tf.identity(tensor_out, name="F0_to_L_out_tensor")
            output_tensor_list[m].append(tensor_out)

            if key is 0:
                if c0x1_params is not None:
                    tensor_out = filter_1_output_1(
                        tensor,
                        rbf,
                        unit_vectors,
                        c0x1_params)
                    # print(key, "x 1 -> 1 tensor_out shapes, T, RBF, O, D", tensor.shape, rbf.shape, tensor_out.shape, output_dim)
                    m = 0 if tensor_out.get_shape().as_list()[-1] == 1 else 1
                    tensor_out = tf.identity(tensor_out, name="F1_to_1_out_tensor")
                    output_tensor_list[m].append(tensor_out)


            if key is 1:
                if c1x1_params is not None:
                    tensor_out = filter_1_output_1(
                        tensor,
                        rbf,
                        unit_vectors,
                        c1x1_params)
                    # print(key, "x 1 -> 1 tensor_out shapes, T, RBF, O, D", tensor.shape, rbf.shape, tensor_out.shape, output_dim)
                    m = 0 if tensor_out.get_shape().as_list()[-1] == 1 else 1
                    tensor_out = tf.identity(tensor_out, name="F1_to_1_out_tensor")
                    output_tensor_list[m].append(tensor_out)

                if c1x0_params is not None:
                    # L x 1 -> 0, dot product of two vectors
                    tensor_out = filter_1_output_0(
                        tensor,
                        rbf,
                        unit_vectors,
                        c1x0_params)
                    # print(key, "x 1 -> 0 tensor_out shapes, T, RBF, O, D", tensor.shape, rbf.shape, tensor_out.shape, output_dim)
                    m = 0 if tensor_out.get_shape().as_list()[-1] == 1 else 1
                    tensor_out = tf.identity(tensor_out, name="F1_to_0_out_tensor")
                    output_tensor_list[m].append(tensor_out)

    return output_tensor_list


def self_interaction(input_tensor_list, ixn_0_w, ixn_0_b, ixn_1_w):
    output_tensor_list = {0: [], 1: []}
    for key in input_tensor_list:
        for i, tensor in enumerate(input_tensor_list[key]):
            if key == 0:
                tensor_out = self_interaction_layer_with_biases(tensor, ixn_0_w, ixn_0_b)
            else:
                if ixn_1_w is not None: # (ytz): None for the last layer
                    tensor_out = self_interaction_layer_without_biases(tensor, ixn_1_w)
            m = 0 if tensor_out.get_shape().as_list()[-1] == 1 else 1
            output_tensor_list[m].append(tensor_out)
    return output_tensor_list

# skip for the last layer
def nonlinearity(input_tensor_list, biases, nonlin=tf.nn.elu):
    with tf.variable_scope(None, "nonlinearity", values=[input_tensor_list]):
        output_tensor_list = {0: [], 1: []}
        for key in input_tensor_list:
            for i, tensor in enumerate(input_tensor_list[key]):
                if key == 0:
                    # assert biases is None
                    tensor_out = nn_utils.rotation_equivariant_nonlinearity(
                        tensor,
                        None,
                        nonlin=nonlin)
                else:
                    # assert biases is not None
                    if biases is not None: # None for the last layer
                        tensor_out = nn_utils.rotation_equivariant_nonlinearity(
                            tensor,
                            biases,
                            nonlin=nonlin)
                m = 0 if tensor_out.get_shape().as_list()[-1] == 1 else 1
                output_tensor_list[m].append(tensor_out)
        return output_tensor_list


def concatenation(input_tensor_list):
    output_tensor_list = {0: [], 1: []}
    for key in input_tensor_list:
        with tf.variable_scope(None, "L" + str(key), values=input_tensor_list[key]):
            # Concatenate along channel axis
            # [N, channels, M]
            output_tensor_list[key].append(tf.concat(input_tensor_list[key], axis=-2))
    return output_tensor_list
