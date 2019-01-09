import numpy as np
import unittest

import tensorflow as tf
from timemachine import observable

class TestObservable(unittest.TestCase):

	def tearDown(self):
		tf.reset_default_graph()

	def test_dense_sorted_dij(self):

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