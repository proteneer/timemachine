from math import sqrt
import tensorflow as tf
import numpy as np
from timemachine import nn_utils
from nn_utils import FLOAT_TYPE, EPSILON

# Layers for 3D rotation-equivariant network.


trainable_vars = {}

def R(inputs, nonlin=tf.nn.relu, hidden_dim=None, output_dim=1, weights_initializer=None, biases_initializer=None):
    with tf.variable_scope(None, "radial_function", values=[inputs]):
        if weights_initializer is None:
            print("WIX", )
            weights_initializer = tf.contrib.layers.xavier_initializer()
        if biases_initializer is None:
            biases_initializer = tf.constant_initializer(0.)
        input_dim = inputs.get_shape()[-1]
        if hidden_dim is None:
            hidden_dim = input_dim

        # needs to be GC'd?
        

        # w1 = tf.get_variable('weights1', [hidden_dim, input_dim], dtype=FLOAT_TYPE, initializer=weights_initializer)
        # b1 = tf.get_variable('biases1', [hidden_dim], dtype=FLOAT_TYPE, initializer=biases_initializer)
        # w2 = tf.get_variable('weights2', [output_dim, hidden_dim], dtype=FLOAT_TYPE, initializer=weights_initializer)
        # b2 = tf.get_variable('biases2', [output_dim], dtype=FLOAT_TYPE, initializer=biases_initializer)

        prefix = tf.get_variable_scope().name
        w1_name = prefix+"/weights1:0"
        w2_name = prefix+"/weights2:0"
        b1_name = prefix+"/biases1:0"
        b2_name = prefix+"/biases2:0"

        if w1_name not in trainable_vars:
            trainable_vars[w1_name] = tf.get_variable('weights1', [hidden_dim, input_dim], dtype=FLOAT_TYPE, initializer=weights_initializer)
        w1 = trainable_vars[w1_name]

        if w2_name not in trainable_vars:
            trainable_vars[w2_name] = tf.get_variable('weights2', [output_dim, hidden_dim], dtype=FLOAT_TYPE, initializer=weights_initializer)
        w2 = trainable_vars[w2_name]

        if b1_name not in trainable_vars:
            trainable_vars[b1_name] = tf.get_variable('biases1', [hidden_dim], dtype=FLOAT_TYPE, initializer=weights_initializer)
        b1 = trainable_vars[b1_name]

        if b2_name not in trainable_vars:
            trainable_vars[b2_name] = tf.get_variable('biases2', [output_dim], dtype=FLOAT_TYPE, initializer=weights_initializer)
        b2 = trainable_vars[b2_name]



        # if w2_name in trainable_vars:
        #     w2 = trainable_vars[w2.name]
        # else:
        #     trainable_vars[w2.name] = w2

        # if b1.name in trainable_vars:
        #     b1 = trainable_vars[b1.name]
        # else:
        #     trainable_vars[b1.name] = b1

        # if b2.name in trainable_vars:
        #     b2 = trainable_vars[b2.name]
        # else:
        #     trainable_vars[b2.name] = b2

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


def F_0(inputs, nonlin=tf.nn.relu, hidden_dim=None, output_dim=1,
        weights_initializer=None, biases_initializer=None):
    # [N, N, output_dim, 1]
    with tf.variable_scope(None, "F_0", values=[inputs]):
        return tf.expand_dims(
            R(inputs, nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim,
              weights_initializer=weights_initializer, biases_initializer=biases_initializer),
            axis=-1)


def F_1(inputs, rij, nonlin=tf.nn.relu, hidden_dim=None, output_dim=1,
        weights_initializer=None, biases_initializer=None):
    with tf.variable_scope(None, "F_1", values=[inputs]):
        # [N, N, output_dim]
        radial = R(inputs, nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim,
                   weights_initializer=weights_initializer, biases_initializer=biases_initializer)
        # Mask out for dij = 0
        dij = tf.norm(rij, axis=-1)
        condition = tf.tile(tf.expand_dims(dij < EPSILON, axis=-1), [1, 1, output_dim])
        masked_radial = tf.where(condition, tf.zeros_like(radial), radial)
        # [N, N, output_dim, 3]
        return tf.expand_dims(unit_vectors(rij), axis=-2) * tf.expand_dims(masked_radial, axis=-1)


def F_2(inputs, rij, nonlin=tf.nn.relu, hidden_dim=None, output_dim=1,
        weights_initializer=None, biases_initializer=None):
    with tf.variable_scope(None, "F_2", values=[inputs]):
        # [N, N, output_dim]
        radial = R(inputs, nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim,
                   weights_initializer=weights_initializer, biases_initializer=biases_initializer)
        # Mask out for dij = 0
        dij = tf.norm(rij, axis=-1)
        condition = tf.tile(tf.expand_dims(dij < EPSILON, axis=-1), [1, 1, output_dim])
        masked_radial = tf.where(condition, tf.zeros_like(radial), radial)
        # [N, N, output_dim, 5]
        return tf.expand_dims(Y_2(rij), axis=-2) * tf.expand_dims(masked_radial, axis=-1)


def filter_0(layer_input,
             rbf_inputs,
             nonlin=tf.nn.relu,
             hidden_dim=None,
             output_dim=1,
             weights_initializer=None,
             biases_initializer=None):
    with tf.variable_scope(None, "F0_to_L", values=[layer_input]):
        # [N, N, output_dim, 1]
        F_0_out = F_0(rbf_inputs, nonlin=nonlin, hidden_dim=hidden_dim, output_dim=output_dim,
                      weights_initializer=weights_initializer, biases_initializer=biases_initializer)
        # [N, output_dim]
        input_dim = layer_input.get_shape().as_list()[-1]
        # print("filter_0 input_dim", input_dim, layer_input.shape)
        # Expand filter axis "j"
        cg = tf.expand_dims(tf.eye(input_dim, dtype=FLOAT_TYPE), axis=-2)
        print("DTYPES", cg.dtype, F_0_out.dtype, layer_input.dtype)
        return tf.einsum('ijk,abfj,bfk->afi', cg, F_0_out, layer_input)

# F x I -> O
# 0 x 0 -> 0 scalar product
# 1 x 0 -> 0 not allowed
# 1 x 1 -> 0 dot product
# 1 x 0 -> 1 vector scalar
# 1 x 1 -> 1 cross product

def filter_1_output_0(layer_input,
                      rbf_inputs,
                      rij,
                      nonlin=tf.nn.relu,
                      hidden_dim=None,
                      output_dim=1,
                      weights_initializer=None,
                      biases_initializer=None):
    with tf.variable_scope(None, "F1_to_0", values=[layer_input]):
        # [N, N, output_dim, 3]
        F_1_out = F_1(rbf_inputs,
                      rij,
                      nonlin=nonlin,
                      hidden_dim=hidden_dim,
                      output_dim=output_dim,
                      weights_initializer=weights_initializer,
                      biases_initializer=biases_initializer)
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
                      nonlin=tf.nn.relu,
                      hidden_dim=None,
                      output_dim=1,
                      weights_initializer=None,
                      biases_initializer=None):
    with tf.variable_scope(None, "F1_to_1", values=[layer_input]):
        # [N, N, output_dim, 3]
        F_1_out = F_1(rbf_inputs,
                      rij,
                      nonlin=nonlin,
                      hidden_dim=hidden_dim,
                      output_dim=output_dim,
                      weights_initializer=weights_initializer,
                      biases_initializer=biases_initializer)
        # [N, output_dim, 3]
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
                      nonlin=tf.nn.relu,
                      hidden_dim=None,
                      output_dim=1,
                      weights_initializer=None,
                      biases_initializer=None):
    with tf.variable_scope(None, "F2_to_2", values=[layer_input]):
        # [N, N, output_dim, 3]
        F_2_out = F_2(rbf_inputs,
                      rij,
                      nonlin=nonlin,
                      hidden_dim=hidden_dim,
                      output_dim=output_dim,
                      weights_initializer=weights_initializer,
                      biases_initializer=biases_initializer)
        # [N, output_dim, 5]
        if layer_input.get_shape().as_list()[-1] == 1:
            # 0 x 2 -> 2
            cg = tf.expand_dims(tf.eye(5), axis=-1)
            return tf.einsum('ijk,abfj,bfk->afi', cg, F_2_out, layer_input)
        else:
            raise NotImplementedError("Other Ls not implemented")


def self_interaction_layer_without_biases(inputs, output_dim, weights_initializer=None, biases_initializer=None):
    # input has shape [N, C, 2L+1]
    # input_dim is number of channels
    if weights_initializer is None:
        weights_initializer = tf.orthogonal_initializer()
    if biases_initializer is None:
        biases_initializer = tf.constant_initializer(0.)

    with tf.variable_scope(None, "self_interaction_layer", values=[inputs]):
        input_dim = inputs.get_shape().as_list()[-2]

        prefix = tf.get_variable_scope().name
        w_si_name = prefix+"/weights:0"

        if w_si_name not in trainable_vars:
            print("initializing..")
            trainable_vars[w_si_name] = tf.get_variable('weights', [output_dim, input_dim], dtype=FLOAT_TYPE, initializer=weights_initializer)
        w_si = trainable_vars[w_si_name]

        # [N, output_dim, 2l+1]
        return tf.transpose(tf.einsum('afi,gf->aig', inputs, w_si), perm=[0, 2, 1])


def self_interaction_layer_with_biases(inputs, output_dim, weights_initializer=None, biases_initializer=None):
    # input has shape [N, C, 2L+1]
    # input_dim is number of channels
    if weights_initializer is None:
        weights_initializer = tf.orthogonal_initializer()
    if biases_initializer is None:
        biases_initializer = tf.constant_initializer(0.)

    with tf.variable_scope(None, "self_interaction_layer", values=[inputs]):
        input_dim = inputs.get_shape().as_list()[-2]

        prefix = tf.get_variable_scope().name
        w_si_name = prefix+"/weights:0"
        b_si_name = prefix+"/biases:0"

        if w_si_name not in trainable_vars:
            trainable_vars[w_si_name] = tf.get_variable('weights', [output_dim, input_dim], dtype=FLOAT_TYPE, initializer=weights_initializer)
        w_si = trainable_vars[w_si_name]

        if b_si_name not in trainable_vars:
            trainable_vars[b_si_name] = tf.get_variable('biases', [output_dim], dtype=FLOAT_TYPE, initializer=biases_initializer)
        b_si = trainable_vars[b_si_name]

        # [N, output_dim, 2l+1]
        return tf.transpose(tf.einsum('afi,gf->aig', inputs, w_si) + b_si, perm=[0, 2, 1])


def convolution(input_tensor_list, rbf, unit_vectors, weights_initializer=None, biases_initializer=None):
    # print("Start Convolution")
    with tf.variable_scope(None, "convolution", values=[input_tensor_list]):
        output_tensor_list = {0: [], 1: []}
        for key in input_tensor_list:
            # print("Key:", key)
            with tf.variable_scope(None, "L" + str(key), values=input_tensor_list[key]):
                for i, tensor in enumerate(input_tensor_list[key]):
                    output_dim = tensor.get_shape().as_list()[-2]
                    with tf.variable_scope(None, 'tensor_' + str(i), values=[tensor]):
                        tensor = tf.identity(tensor, name="in_tensor")
                        if True:
                            # L x 0 -> L

                            tensor_out = filter_0(tensor,
                                                  rbf,
                                                  output_dim=output_dim,
                                                  weights_initializer=weights_initializer,
                                                  biases_initializer=biases_initializer)
                            # print(key, "x 0 -> L tensor_out shapes, T, RBF, O, D", tensor.shape, rbf.shape, tensor_out.shape, output_dim)
                            m = 0 if tensor_out.get_shape().as_list()[-1] == 1 else 1
                            tensor_out = tf.identity(tensor_out, name="F0_to_L_out_tensor")
                            output_tensor_list[m].append(tensor_out)
                        if key is 1:
                            # L x 1 -> 0
                            tensor_out = filter_1_output_0(tensor,
                                                           rbf,
                                                           unit_vectors,
                                                           output_dim=output_dim,
                                                           weights_initializer=weights_initializer,
                                                           biases_initializer=biases_initializer)
                            # print(key, "x 1 -> 0 tensor_out shapes, T, RBF, O, D", tensor.shape, rbf.shape, tensor_out.shape, output_dim)
                            m = 0 if tensor_out.get_shape().as_list()[-1] == 1 else 1
                            tensor_out = tf.identity(tensor_out, name="F1_to_0_out_tensor")
                            output_tensor_list[m].append(tensor_out)



                        if key is 0 or key is 1:
                            # L x 1 -> 1
                            tensor_out = filter_1_output_1(tensor,
                                                           rbf,
                                                           unit_vectors,
                                                           output_dim=output_dim,
                                                           weights_initializer=weights_initializer,
                                                           biases_initializer=biases_initializer)
                            # print(key, "x 1 -> 1 tensor_out shapes, T, RBF, O, D", tensor.shape, rbf.shape, tensor_out.shape, output_dim)
                            m = 0 if tensor_out.get_shape().as_list()[-1] == 1 else 1
                            tensor_out = tf.identity(tensor_out, name="F1_to_1_out_tensor")
                            output_tensor_list[m].append(tensor_out)
        return output_tensor_list


def self_interaction(input_tensor_list, output_dim, weights_initializer=None, biases_initializer=None):
    with tf.variable_scope(None, "self_interaction", values=[input_tensor_list]):
        output_tensor_list = {0: [], 1: []}
        for key in input_tensor_list:
            with tf.variable_scope(None, "L" + str(key), values=input_tensor_list[key]):
                for i, tensor in enumerate(input_tensor_list[key]):
                    with tf.variable_scope(None, 'tensor_' + str(i), values=[tensor]):
                        if key == 0:
                            tensor_out = self_interaction_layer_with_biases(tensor,
                                                                            output_dim,
                                                                            weights_initializer=weights_initializer,
                                                                            biases_initializer=biases_initializer)
                        else:
                            tensor_out = self_interaction_layer_without_biases(tensor,
                                                                               output_dim,
                                                                               weights_initializer=weights_initializer,
                                                                               biases_initializer=biases_initializer)
                        m = 0 if tensor_out.get_shape().as_list()[-1] == 1 else 1
                        output_tensor_list[m].append(tensor_out)
        return output_tensor_list


def nonlinearity(input_tensor_list, nonlin=tf.nn.elu, biases_initializer=None):
    with tf.variable_scope(None, "nonlinearity", values=[input_tensor_list]):
        output_tensor_list = {0: [], 1: []}
        for key in input_tensor_list:
            with tf.variable_scope(None, "L" + str(key), values=input_tensor_list[key]):
                for i, tensor in enumerate(input_tensor_list[key]):
                    with tf.variable_scope(None, 'tensor_' + str(i), values=[tensor]):
                        if key == 0:
                            tensor_out = nn_utils.rotation_equivariant_nonlinearity(tensor,
                                                                                 nonlin=nonlin,
                                                                                 biases_initializer=biases_initializer)
                        else:
                            tensor_out = nn_utils.rotation_equivariant_nonlinearity(tensor,
                                                                                 nonlin=nonlin,
                                                                                 biases_initializer=biases_initializer)
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
