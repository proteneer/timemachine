import numpy as np
import tensorflow as tf
import unittest

from timemachine import bonded_force
from timemachine.constants import BOLTZ
from timemachine import integrator
from tensorflow.python.ops.parallel_for.gradients import jacobian

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
        num_atoms = len(self.invMasses)
        num_dims = 3

        if self.v_t is None:
            self.v_t = np.zeros((num_atoms, num_dims))

        noise = self.normal.sample((num_atoms, num_dims))
        noise = tf.cast(noise, dtype=grads.dtype)

        if self.disable_noise:
            noise = tf.zeros(noise.shape, dtype=grads.dtype)

        # (ytz): * operator isn't defined for sparse grads (resulting from tf.gather ops), hence the tf.multiply
        self.v_t = self.vscale*self.v_t - tf.multiply(self.fscale*self.invMasses, grads) + self.nscale*self.sqrtInvMasses*noise
        dx = self.v_t * self.dt
        return dx


class TestLangevinIntegrator(unittest.TestCase):


    def tearDown(self):
        # (ytz): needed to clear variables
        tf.reset_default_graph()

    def setUp(self):
        self.masses = np.array([1.0, 12.0, 4.0])
        self.x0 = np.array([
            [1.0, 0.5, -0.5],
            [0.2, 0.1, -0.3],
            [0.5, 0.4, 0.3],
        ], dtype=np.float64)
        self.x0.setflags(write=False)

        bond_params = [
            tf.get_variable("HO_kb", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(100.0)),
            tf.get_variable("HO_b0", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(2.0)),
        ]

        bond_idxs = np.array([
            [0, 1],
            [1, 2]
        ], dtype=np.int32)

        param_idxs = np.array([
            [0, 1],
            [0, 1],
        ])

        self.hb = bonded_force.HarmonicBondForce(
            bond_params,
            bond_idxs,
            param_idxs,
        )

        angle_params = [
            tf.get_variable("HCH_ka", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(np.sqrt(75.0))),
            tf.get_variable("HCH_a0", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(1.81)),
        ]

        self.ha = bonded_force.HarmonicAngleForce(
            params=angle_params,
            angle_idxs=np.array([[1,0,2]], dtype=np.int32),
            param_idxs=np.array([[0,1]], dtype=np.int32)
        )

    def test_converged_zetas(self):
        """
        Testing convergence of zetas.
        """

        friction = 10.0
        dt = 0.08
        temp = 0.0
        num_atoms = self.x0.shape[0]

        x_ph = tf.placeholder(dtype=tf.float64, shape=(num_atoms, 3))

        num_steps = 100 # with a temp of 10.0 should converge really quickly

        with tf.variable_scope("reference"):
            ref_intg = integrator.LangevinIntegrator(
                self.masses, x_ph, [self.ha, self.hb], dt, friction, temp)
            # ref_intg.vscale = 0.45 -> so we should converge fully to 16 decimals after 47 steps

        with tf.variable_scope("test"):
            test_intg = integrator.LangevinIntegrator(
                self.masses, x_ph, [self.ha, self.hb], dt, friction, temp, buffer_size=50)

        ref_dx, ref_dxdps = ref_intg.step_op()
        test_dx, test_dxdps = test_intg.step_op()

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        x_ref = np.copy(self.x0)
        x_test = np.copy(self.x0)
        for step in range(num_steps):

            ref_dx_val, ref_dxdp_val = sess.run([ref_dx, ref_dxdps], feed_dict={x_ph: x_ref})
            test_dx_val, test_dxdp_val = sess.run([test_dx, test_dxdps], feed_dict={x_ph: x_test})

            np.testing.assert_array_almost_equal(ref_dx_val, test_dx_val, decimal=14)
            np.testing.assert_array_almost_equal(ref_dxdp_val, test_dxdp_val, decimal=14) # BAD WTF CONVERGENCE

            x_ref += ref_dx_val
            x_test += test_dx_val


    def test_ten_steps(self):
        """
        Testing against reference implementation.
        """
        friction = 10.0
        dt = 0.003
        temp = 0.0
        num_atoms = len(self.masses)
        x_ph = tf.placeholder(dtype=tf.float64, shape=(num_atoms, 3))

        hb = self.hb
        ha = self.ha

        ref_intg = ReferenceLangevinIntegrator(self.masses, dt, friction, temp)

        num_steps = 10

        x = x_ph

        for step in range(num_steps):
            all_grads = []
            for force in [self.hb, self.ha]:
                all_grads.append(force.gradients(x))
            all_grads = tf.stack(all_grads, axis=0)
            grads = tf.reduce_sum(all_grads, axis=0)
            dx = ref_intg.step(grads)
            x += dx

        ref_x_final_op = x

        # verify correctness of jacobians through time
        ref_dxdp_hb_op = jacobian(x, hb.get_params(), use_pfor=False)
        ref_dxdp_ha_op = jacobian(x, ha.get_params(), use_pfor=False)

        test_intg = integrator.LangevinIntegrator(self.masses, x_ph, [hb, ha], dt, friction, temp)
        dx_op, dxdps_op = test_intg.step_op()
        # dxdps_op = tf.reduce_sum(dxdps_op, axis=[1,2])

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        ref_x_final, ref_dxdp_hb, ref_dxdp_ha = sess.run([ref_x_final_op, ref_dxdp_hb_op, ref_dxdp_ha_op], feed_dict={x_ph: self.x0})

        x = np.copy(self.x0) # this copy is super important else it just modifies everything in place
        for step in range(num_steps):
            dx_val, dxdp_val = sess.run([dx_op, dxdps_op], feed_dict={x_ph: x})
            x += dx_val
        test_dxdp = dxdp_val
        test_x_final_val = x

        np.testing.assert_array_almost_equal(ref_x_final, test_x_final_val, decimal=14)
        np.testing.assert_array_almost_equal(np.concatenate([ref_dxdp_hb, ref_dxdp_ha]), test_dxdp, decimal=14) # BAD, restore to 13

        # test grads_and_vars and computation of higher derivatives
        x_opt = np.array([
            [-0.0070, -0.0100, 0.0000],
            [-0.1604,  0.4921, 0.0000],
            [ 0.5175,  0.0128, 0.0000],
        ], dtype=np.float64) # idealized geometry

        def loss(pred_x):

            # Compute pairwise distances
            def dij(x):
                v01 = x[0]-x[1]
                v02 = x[0]-x[2]
                v12 = x[1]-x[2]
                return tf.stack([tf.norm(v01), tf.norm(v02), tf.norm(v12)])

            return tf.norm(dij(x_opt) - dij(pred_x))

        x_final_ph = tf.placeholder(dtype=tf.float64, shape=(num_atoms, 3))

        l0 = loss(ref_x_final_op)
        l1 = loss(x_final_ph)

        ref_dLdp_op = tf.gradients(l0, self.hb.params+self.ha.params) # goes through reference integrator
        test_dLdx_op = tf.gradients(l1, x_final_ph)
        test_dLdp_op_gvs = test_intg.grads_and_vars(test_dLdx_op[0]) # multiply with dxdp

        # need to fix this test. 
        ref_dLdp = sess.run(ref_dLdp_op, feed_dict={x_ph: self.x0})
        test_dLdp = sess.run([a[0] for a in test_dLdp_op_gvs], feed_dict={x_final_ph: test_x_final_val})

        np.testing.assert_array_almost_equal(ref_dLdp, test_dLdp)


if __name__ == "__main__":

    unittest.main()
