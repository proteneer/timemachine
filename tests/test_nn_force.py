import scipy
import unittest
import numpy as np
import tensorflow as tf
from timemachine.functionals.nn import Tensorfield
from timemachine import derivatives


def random_rotation_matrix():
    """
    Generates a random 3D rotation matrix from axis and angle.

    Returns:
        Random rotation matrix.
    """
    rng = np.random.RandomState()
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis)    
    # axis /= np.linalg.norm(axis) + EPSILON
    theta = 2 * np.pi * rng.uniform(0.0, 1.0)
    return rotation_matrix(axis, theta)


def rotation_matrix(axis, theta):
    return scipy.linalg.expm(np.cross(np.eye(3), axis * theta))


class TestTensorfield(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_tensorfield_force(self):
        """
        Testing an implementation of the nn force
        """
        x0 = np.array([
            [ 0.0637,   0.0126,   0.2203],
            [ 1.0573,  -0.2011,   1.2864],
            [ 2.3928,   1.2209,  -0.2230],
            [-0.6891,   1.6983,   0.0780],
            [-0.6312,  -1.6261,  -0.2601]
        ], dtype=np.float64)

        num_rbfs = 4
        x_ph = tf.placeholder(shape=(5, 3), dtype=np.float64)
        compressed_atom_types = np.array([1,0,0,0,0], dtype=np.int32)
        num_types = 2
        # layer_dims = [11,17,31,44,1]
        layer_dims = [4,9,13,1]

        params = [
            tf.get_variable('i_self_ixn_w', [layer_dims[0], num_types], dtype=tf.float64, initializer=tf.orthogonal_initializer()),
            tf.get_variable('i_self_ixn_b', [layer_dims[0]], dtype=tf.float64, initializer=tf.constant_initializer(0.))
        ]

        self_ixn_param_idxs = [0, 1]
        layer_param_idxs = []

        def is_first_layer(l_idx):
            return l_idx == 0

        def is_last_layer(l_idx):
            return l_idx == len(layer_dims)-2

        for l_idx, l in enumerate(layer_dims):

            if l_idx == len(layer_dims) - 1:
                break

            # generate convolutional variables
            all_conv_idxs = [] # slot 0

            for edge in range(4):

                if is_first_layer(l_idx) and edge > 1:
                    all_conv_idxs.append(None)
                    continue

                if is_last_layer(l_idx):
                    if edge == 1 or edge == 3:
                        all_conv_idxs.append(None)
                        continue

                input_dim = num_rbfs
                hidden_dim = input_dim
                output_dim = l

                prefix = str(l_idx)+'_'+str(edge)+'_'

                w0_idx = len(params)
                w0 = params.append(tf.get_variable(prefix+'conv_r_w0', [hidden_dim, input_dim], dtype=tf.float64, initializer=tf.orthogonal_initializer()))
                b0_idx = len(params)
                b0 = params.append(tf.get_variable(prefix+'conv_r_b0', [hidden_dim], dtype=tf.float64, initializer=tf.constant_initializer(0.)))


                w1_idx = len(params)
                w1 = params.append(tf.get_variable(prefix+'conv_r_w1', [output_dim, hidden_dim], dtype=tf.float64, initializer=tf.orthogonal_initializer()))
                b1_idx = len(params)
                b1 = params.append(tf.get_variable(prefix+'conv_r_b1', [output_dim], dtype=tf.float64, initializer=tf.constant_initializer(0.)))

                all_conv_idxs.append([w0_idx, b0_idx, w1_idx, b1_idx])

            # if l_idx == 0:
                # all_conv_idxs[2] = None
                # all_conv_idxs[3] = None

            next_dim = layer_dims[l_idx + 1]

            if l_idx == 0:
                d0 = l
                d1 = l
            else:
                d0 = l*2
                d1 = l*3

            ixn_w0_idx = len(params)
            params.append(tf.get_variable(str(l_idx)+'_self_ixn_w0', [next_dim, d0], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer()))
            ixn_b0_idx = len(params)
            params.append(tf.get_variable(str(l_idx)+'_self_ixn_b0', [next_dim], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer()))

            if is_last_layer(l_idx):
                ixn_w1_idx = None
                nonlin_idx = None
            else:   
                ixn_w1_idx = len(params)
                params.append(tf.get_variable(str(l_idx)+'_self_ixn_w1', [next_dim, d1], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer()))
                nonlin_idx = len(params) # used for l>1 orders, eg. 5, 7, 9
                params.append(tf.get_variable(str(l_idx)+'_nonlin_', [next_dim], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer()))

            layer_param_idxs.append([
                all_conv_idxs, # [0]
                [ixn_w0_idx, ixn_b0_idx, ixn_w1_idx], #[1]
                nonlin_idx, #[2]
            ])

        rbf_low = 0.
        rbf_high = 4.6
        rbf_count = 4
        rbf_spacing = (rbf_high - rbf_low) / rbf_count
        rbf_centers = tf.cast(tf.lin_space(rbf_low, rbf_high, rbf_count), tf.float64)

        tff = Tensorfield(
            params,
            layer_param_idxs,
            self_ixn_param_idxs,
            compressed_atom_types,
            rbf_centers,
            rbf_spacing)

        # test equivariance
        test_energy_op = tff.energy(x_ph)

        test_grads_op, test_hessians_op, test_mixed_op = derivatives.compute_ghm(test_energy_op, x_ph, tff.get_params())

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        ref_e, ref_g, ref_h, ref_m = sess.run([
            test_energy_op,
            test_grads_op,
            test_hessians_op,
            test_mixed_op], feed_dict={x_ph: x0})

        assert not np.any(np.isnan(ref_e))
        assert not np.any(np.isnan(ref_g))
        assert not np.any(np.isnan(ref_h))

        for dd in ref_m:
            assert not np.any(np.isnan(dd))

        for _ in range(10):
            rotation = random_rotation_matrix()
            rotated_shape = np.dot(x0, rotation)
            translation = np.expand_dims(np.random.uniform(low=-3., high=3., size=(3)), axis=0)
            translated_shape = rotated_shape + translation

            rot_e, rot_g, rot_h, rot_m = sess.run([
                test_energy_op,
                test_grads_op,
                test_hessians_op,
                test_mixed_op], feed_dict={x_ph: translated_shape})

            np.testing.assert_array_almost_equal(ref_e, rot_e)
            # we expect gradients to also be rotated by the same amount, but the net
            # translational component should be zero
            np.testing.assert_array_almost_equal(np.dot(ref_g, rotation), rot_g)

            # TODO: need to add equivariance tests for hessians and mixed partials

if __name__ == "__main__":
    unittest.main()