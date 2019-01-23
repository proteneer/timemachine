import numpy as np
import unittest

import tensorflow as tf
from timemachine import observable
# from timemachine.functionals import bonded
from timemachine.constants import VIBRATIONAL_CONSTANT


class TestObservable(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    # def test_vibrational_frequencies(self):
    #     # conf, energies):
    #     x_opt = np.array([
    #         [-0.0070, -0.0100, 0.0000],
    #         [-0.1604,  0.4921, 0.0000],
    #         [ 0.5175,  0.0128, 0.0000],
    #     ], dtype=np.float64) # idealized geometry

    #     bond_params = [
    #         tf.get_variable("HO_kb", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(100.0)),
    #         tf.get_variable("HO_b0", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(0.51)),
    #     ]

    #     hb = bonded_force.HarmonicBondForce(
    #         params=bond_params,
    #         bond_idxs=np.array([[0,1],[0,2]], dtype=np.int32),
    #         param_idxs=np.array([[0,1],[0,1]], dtype=np.int32)
    #     )

    #     angle_params = [
    #         tf.get_variable("HOH_ka", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(75.0)),
    #         tf.get_variable("HOH_a0", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(1.81)),
    #     ]

    #     ha = bonded_force.HarmonicAngleForce(
    #         params=angle_params,
    #         angle_idxs=np.array([[1,0,2]], dtype=np.int32),
    #         param_idxs=np.array([[0,1]], dtype=np.int32)
    #     )

    #     x_ph = tf.placeholder(shape=(3, 3), dtype=np.float64)


    #     test_eigs = observable.vibrational_eigenvalues(x_ph, np.array([8.0, 1.0, 1.0], dtype=np.float64), [hb, ha])

    #     true_freqs = [0,0,0,40.63,59.383,66.44,1799.2,3809.46,3943] # from http://gaussian.com/vib/
    #     true_eigs = [(x/VIBRATIONAL_CONSTANT)**2 for x in true_freqs]
    #     loss = tf.pow(true_eigs - test_eigs, 2)
    #     dfdp = tf.gradients(loss, bond_params+angle_params)
    #     print(loss)
    #     sess = tf.Session()
    #     sess.run(tf.initializers.global_variables())
    #     print(sess.run([loss, dfdp], feed_dict={x_ph: x_opt}))

    def test_dense_sorted_dij(self):
        """
        Testing sorted distance matrix observables implemented correctly
        """
        x0 = np.array([
            [-0.0070, -0.0100, 0.0000],
            [-0.1604,  0.4921, 0.0000],
            [ 0.5175,  0.0128, 0.0000],
        ], dtype=np.float64)

        all_xs = []
        n_confs = 5
        for p in range(n_confs):
            rand_dx = np.random.rand(3, 3)/10
            all_xs.append(x0 + rand_dx)

        stacked_dijs = []

        for x in all_xs:
            # loop over the atoms
            dij = np.zeros(shape=(3,3), dtype=np.float64)
            for a_idx, a in enumerate(x):
                for b_idx, b in enumerate(x):
                    dij[a_idx][b_idx] = np.sum(np.power(a-b, 2), axis=-1)
            stacked_dijs.append(dij)

        stacked_dijs = np.stack(stacked_dijs, axis=-1)
        reference_sorted_dijs = np.sort(stacked_dijs, axis=-1)

        x_ph = tf.placeholder(shape=(n_confs, 3, 3), dtype=np.float64)

        test_sorted_dijs_op = observable.sorted_squared_distances(x_ph)
        dOdx_op = tf.gradients(test_sorted_dijs_op, x_ph)

        sess = tf.Session()
        test_sorted_dijs, test_grads = sess.run([test_sorted_dijs_op, dOdx_op], feed_dict={x_ph: np.array(all_xs, dtype=np.float64)})

        # test that distances are equivalent
        np.testing.assert_almost_equal(
            reference_sorted_dijs,
            test_sorted_dijs,
            decimal=15)

        assert test_grads is not None
        assert not np.any(np.isnan(test_grads))


if __name__ == "__main__":
    unittest.main()