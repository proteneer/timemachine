import unittest
import numpy as np
import tensorflow as tf
from timemachine.nn_force import TensorfieldForce

class TestTensorfieldForce(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_tensorfield_force(self):
        x0 = np.array([
            [ 0.0637,   0.0126,   0.2203],
            [ 1.0573,  -0.2011,   1.2864],
            [ 2.3928,   1.2209,  -0.2230],
            [-0.6891,   1.6983,   0.0780],
            [-0.6312,  -1.6261,  -0.2601]
        ], dtype=np.float64)

        x_ph = tf.placeholder(shape=(5, 3), dtype=np.float64)

        compressed_atom_types = np.array([1,0,0,0,0], dtype=np.int32)

        tff = TensorfieldForce(compressed_atom_types) # add ability t
        test_energy_op, debug = tff.energy(x_ph)
        # test_grads_op = tff.gradients(x_ph)
        # test_hessians_op = tff.hessians(x_ph)

        opt = tf.train.AdamOptimizer()

        loss = tf.pow(10.0 - test_energy_op, 2)

        train_opt = opt.minimize(loss)


        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        # energy = 10.0

        for _ in range(100):
            _, res = sess.run([train_opt, loss], feed_dict={x_ph: x0})
            print(res)
            # assert 0

        # print(sess.run([debug[0][0], debug[1][0]],  feed_dict={x_ph: x0}))

        # print(sess.run([test_energy_op, test_grads_op, test_hessians_op], feed_dict={x_ph: x0}))

        # test_mixed_op = tff.mixed_partials(x_ph)

if __name__ == "__main__":
    unittest.main()