import numpy as np
import tensorflow as tf
import unittest

from timemachine import force
from timemachine.constants import BOLTZ
from timemachine import integrator

class TestOptimization(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_methane(self):
        """
        Testing minimization of methane to forcefield geometries.
        """
        masses = np.array([6.0, 1.0, 1.0, 1.0, 1.0])
        x0 = np.array([
            [ 0.0637,   0.0126,   0.2203],
            [ 1.0573,  -0.2011,   1.2864],
            [ 2.3928,   1.2209,  -0.2230],
            [-0.6891,   1.6983,   0.0780],
            [-0.6312,  -1.6261,  -0.2601]
        ], dtype=np.float64)

        num_atoms = len(masses)

        bond_params = [
            tf.get_variable("HH_kb", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(100.0)),
            tf.get_variable("HH_b0", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(1.0)),
        ]

        hb = force.HarmonicBondForce(
            params=bond_params,
            bond_idxs=np.array([[0,1],[0,2],[0,3],[0,4]], dtype=np.int32),
            param_idxs=np.array([[0,1],[0,1],[0,1],[0,1]], dtype=np.int32)
        )

        ideal_angle = 1.9111355

        angle_params = [
            tf.get_variable("HCH_ka", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(75.0)),
            tf.get_variable("HCH_a0", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(ideal_angle)),
        ]

        ha = force.HarmonicAngleForce(
            params=angle_params,
            angle_idxs=np.array([[1,0,2],[1,0,3],[1,0,4],[2,0,3],[2,0,4],[3,0,4]], dtype=np.int32),
            param_idxs=np.array([[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]], dtype=np.int32)
        )

        friction = 10.0
        dt = 0.005
        temp = 300.0

        intg = integrator.LangevinIntegrator(
            masses, [hb, ha], dt, friction, temp, disable_noise=True)

        x_ph = tf.placeholder(dtype=tf.float64, shape=(num_atoms, 3))
        dx_op, dxdp_op = intg.step(x_ph)

        num_steps = 1000

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        x = x0

        for step in range(num_steps):
            dx_val, dxdp_val = sess.run([dx_op, dxdp_op], feed_dict={x_ph: x})
            x += dx_val

        # test idealized bond distances
        bonds = x - x[0, :]
        bond_lengths = np.linalg.norm(bonds[1:, :], axis=1)
        np.testing.assert_almost_equal(bond_lengths, np.array([1.0]*4), decimal=4)

        cj = np.take(x, ha.angle_idxs[:, 0], axis=0)
        ci = np.take(x, ha.angle_idxs[:, 1], axis=0)
        ck = np.take(x, ha.angle_idxs[:, 2], axis=0)
        vij = cj - ci
        vik = ck - ci

        top = np.sum(vij * vik, -1)
        bot = np.linalg.norm(vij, axis=-1)*np.linalg.norm(vik, axis=-1)
        angles = np.arccos(top/bot)

        np.testing.assert_almost_equal(angles, np.array([ideal_angle]*6), decimal=2) # poor precision beause of 0.98
        # test idealized bond angles

