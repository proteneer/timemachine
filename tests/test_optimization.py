import numpy as np
import tensorflow as tf
import unittest

from timemachine import bonded_force
from timemachine.constants import BOLTZ
from timemachine import integrator

class TestOptimization(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_water(self):
        """
        Testing optimization of bonded_forcefield parameters so that an non-equilibriated OH2 can minimize into the correct condensed phase angle.
        """
        masses = np.array([8.0, 1.0, 1.0])
        x0 = np.array([
            [-0.0070, -0.0100, 0.0000],
            [-1.1426,  0.5814, 0.0000],
            [ 0.4728, -0.2997, 0.0000],
        ], dtype=np.float64) # starting geometry

        x_opt = np.array([
            [-0.0070, -0.0100, 0.0000],
            [-0.1604,  0.4921, 0.0000],
            [ 0.5175,  0.0128, 0.0000],
        ], dtype=np.float64) # idealized geometry

        bonds = x_opt - x_opt[0, :]
        bond_lengths = np.linalg.norm(bonds[1:, :], axis=1)

        num_atoms = len(masses)

        starting_bond = 0.8 # Guessestimate starting (true x_opt: 0.52)
        starting_angle = 2.1 # Guessestimate ending (true x_opt: 1.81)

        bond_params = [
            tf.get_variable("OH_kb", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(100.0)),
            tf.get_variable("OH_b0", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(starting_bond)),
        ]

        hb = bonded_force.HarmonicBondForce(
            params=bond_params,
            bond_idxs=np.array([[0,1],[0,2]], dtype=np.int32),
            param_idxs=np.array([[0,1],[0,1]], dtype=np.int32)
        )

        angle_params = [
            tf.get_variable("HOH_ka", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(75.0)),
            tf.get_variable("HOH_a0", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(starting_angle)),
        ]

        ha = bonded_force.HarmonicAngleForce(
            params=angle_params,
            angle_idxs=np.array([[1,0,2]], dtype=np.int32),
            param_idxs=np.array([[0,1]], dtype=np.int32)
        )

        friction = 10.0
        dt = 0.005
        temp = 300.0

        x_ph = tf.placeholder(name="input_geom", dtype=tf.float64, shape=(num_atoms, 3))
        intg = integrator.LangevinIntegrator(
            masses, x_ph, [hb, ha], dt, friction, temp, disable_noise=True, buffer_size=400)

        dx_op, dxdp_op = intg.step_op()

        num_steps = 500

        param_optimizer = tf.train.AdamOptimizer(0.02)

        def loss(pred_x):

            # Compute pairwise distances
            def dij(x):
                v01 = x[0]-x[1]
                v02 = x[0]-x[2]
                v12 = x[1]-x[2]
                return tf.stack([tf.norm(v01), tf.norm(v02), tf.norm(v12)])

            return tf.norm(dij(x_opt) - dij(pred_x))

        # geometry we arrive at at time t=inf
        x_final_ph = tf.placeholder(dtype=tf.float64, shape=(num_atoms, 3))
        dLdx = tf.gradients(loss(x_final_ph), x_final_ph)

        grads_and_vars = intg.grads_and_vars(dLdx)
        train_op = param_optimizer.apply_gradients(grads_and_vars)

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        num_epochs = 75

        for e in range(num_epochs):
            print("starting epoch", e, "current params", sess.run(bond_params+angle_params))
            x = x0  
            intg.reset(sess) # clear integration buffers
            for step in range(num_steps):
                dx_val, dxdp_val = sess.run([dx_op, dxdp_op], feed_dict={x_ph: x})
                x += dx_val

            sess.run(train_op, feed_dict={x_final_ph: x})

        params = sess.run(bond_params+angle_params)
        np.testing.assert_almost_equal(params[1], 0.52, decimal=2)
        np.testing.assert_almost_equal(params[3], 1.81, decimal=1)
           
if __name__ == "__main__":
    unittest.main()