import numpy as np
import tensorflow as tf
import unittest

from timemachine import force
from timemachine.constants import BOLTZ
from timemachine import integrator

# static long long estimate_steps_to_converge(float coeff_a) {
#   double epsilon = 1e-8;
#   return static_cast<long long>(log(epsilon)/log(coeff_a)+1);
# };

# static long long estimate_steps_to_converge(double coeff_a) {
#   double epsilon = 1e-16;
#   return static_cast<long long>(log(epsilon)/log(coeff_a)+1);
# };

class ReferenceLangevinIntegrator():

    def __init__(self, masses, dt=0.0025, friction=1.0, temp=300.0, disable_noise=False):
        self.dt = dt
        self.v_t = None
        self.friction = friction # dissipation speed (how fast we forget)
        self.temperature = temp           # temperature

        self.disable_noise = disable_noise
        self.vscale = np.exp(-self.dt*self.friction)

        if self.friction == 0:
            self.fscale = self.dt
        else:
            self.fscale = (1-self.vscale)/self.friction
        kT = BOLTZ * self.temperature
        self.nscale = np.sqrt(kT*(1-self.vscale*self.vscale)) # noise scale
        self.normal = tf.distributions.Normal(loc=0.0, scale=1.0)
        self.invMasses = (1.0/masses).reshape((-1, 1))
        self.sqrtInvMasses = np.sqrt(self.invMasses)

    def step(self, grads):
        num_atoms = grads.shape[0]
        num_dims = grads.shape[1]

        if self.v_t is None:
            self.v_t = np.zeros((num_atoms, num_dims))

        noise = self.normal.sample((num_atoms, num_dims))
        noise = tf.cast(noise, dtype=grads.dtype)

        if self.disable_noise:
            noise = tf.zeros(noise.shape, dtype=grads.dtype)

        # grads shape: [N x 3]


        self.v_t = self.vscale*self.v_t - self.fscale*self.invMasses*grads + self.nscale*self.sqrtInvMasses*noise
        dx = self.v_t * self.dt
        return dx


class TestLangevinIntegrator(unittest.TestCase):


    def tearDown(self):
        # (ytz): needed to clear variables
        tf.reset_default_graph()

    def test_converged_zetas(self):
        """
        Unittest for ensuring that our converged buffers are working correctly.
        """
        masses = np.array([1.0, 12.0])
        friction = 10.0
        dt = 0.08
        temp = 300.0
        x_ph = tf.placeholder(dtype=tf.float64, shape=(2, 3))
        x0 = np.array([
            [1.0, 0.5, -0.5],
            [0.2, 0.1, -0.3]
        ], dtype=np.float64)

        hb = force.HarmonicBondForce()

        num_steps = 100 # with a temp of 10.0 should converge really quickly

        with tf.variable_scope("reference"):
            ref_intg = integrator.LangevinIntegrator(
                masses, len(hb.params()), dt, friction, temp, disable_noise=True)
            # ref_intg.vscale = 0.45 -> so we should converge fully to 16 decimals after 47 steps

        with tf.variable_scope("test"):
            test_intg = integrator.LangevinIntegrator(
                masses, len(hb.params()), dt, friction, temp, disable_noise=True, buffer_size=50)
            # , buffer_size=50)

        ref_dx, ref_dxdps = ref_intg.step(x_ph, [hb])
        test_dx, test_dxdps = test_intg.step(x_ph, [hb])

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        x_ref = x0
        x_test = x0
        for step in range(num_steps):

            ref_dx_val, ref_dxdp_val = sess.run([ref_dx, ref_dxdps], feed_dict={x_ph: x_ref})
            test_dx_val, test_dxdp_val = sess.run([test_dx, test_dxdps], feed_dict={x_ph: x_test})

            np.testing.assert_array_almost_equal(ref_dx_val, test_dx_val, decimal=14)
            np.testing.assert_array_almost_equal(ref_dxdp_val, test_dxdp_val, decimal=14)

            x_ref += ref_dx_val
            x_test += test_dx_val



        

    def test_five_steps(self):
        """
        Unit test for ensuring that the reference implementation works.
        """
        masses = np.array([1.0, 12.0])
        friction = 10.0
        dt = 0.003
        temp = 300.0
        x_ph = tf.placeholder(dtype=tf.float64, shape=(2, 3))
        x0 = np.array([
            [1.0, 0.5, -0.5],
            [0.2, 0.1, -0.3]
        ], dtype=np.float64)

        hb = force.HarmonicBondForce()
        ref_intg = ReferenceLangevinIntegrator(masses, dt, friction, temp, disable_noise=True)

        num_steps = 4

        x = x_ph

        for step in range(num_steps):
            grads = hb.gradients(x)
            dx = ref_intg.step(grads)
            x += dx

        ref_dxdp = tf.gradients(x, hb.params())
        test_intg = integrator.LangevinIntegrator(masses, len(hb.params()), dt, friction, temp, disable_noise=True)

        dx, dxdps = test_intg.step(x_ph, [hb])
        dxdps = tf.reduce_sum(dxdps, axis=[1,2])

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        obs_x, obs_dxdp = sess.run([x, ref_dxdp], feed_dict={x_ph: x0})

        x = x0

        for step in range(num_steps):
            dx_val, dxdp_val = sess.run([dx, dxdps], feed_dict={x_ph: x})
            x += dx_val

        test_dxdp = dxdp_val
        test_x = x

        np.testing.assert_array_almost_equal(obs_x, test_x, decimal=14)
        np.testing.assert_array_almost_equal(obs_dxdp, test_dxdp, decimal=14)


if __name__ == "__main__":

    unittest.main()
