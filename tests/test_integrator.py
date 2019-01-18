import numpy as np
import tensorflow as tf
import unittest

from timemachine import bonded_force
from timemachine.constants import BOLTZ
from timemachine import integrator
from tensorflow.python.ops.parallel_for.gradients import jacobian

from timemachine.nonbonded_force import LeonnardJones, Electrostatic

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

    def test_converged_zetas(self):
        """
        Testing convergence of zetas.
        """

        masses = np.array([1.0, 12.0, 4.0])
        x0 = np.array([
            [1.0, 0.5, -0.5],
            [0.2, 0.1, -0.3],
            [0.5, 0.4, 0.3],
        ], dtype=np.float64)
        x0.setflags(write=False)

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

        hb = bonded_force.HarmonicBond(
            bond_params,
            bond_idxs,
            param_idxs,
        )

        angle_params = [
            tf.get_variable("HCH_ka", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(np.sqrt(75.0))),
            tf.get_variable("HCH_a0", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(1.81)),
        ]

        ha = bonded_force.HarmonicAngle(
            params=angle_params,
            angle_idxs=np.array([[1,0,2]], dtype=np.int32),
            param_idxs=np.array([[0,1]], dtype=np.int32)
        )

        friction = 10.0
        dt = 0.08
        temp = 0.0
        num_atoms = x0.shape[0]

        x_ph = tf.placeholder(dtype=tf.float64, shape=(num_atoms, 3))

        num_steps = 100 # with a temp of 10.0 should converge really quickly

        with tf.variable_scope("reference"):
            ref_intg = integrator.LangevinIntegrator(
                masses, x_ph, None, [ha, hb], dt, friction, temp)
            # ref_intg.vscale = 0.45 -> so we should converge fully to 16 decimals after 47 steps

        with tf.variable_scope("test"):
            test_intg = integrator.LangevinIntegrator(
                masses, x_ph, None, [ha, hb], dt, friction, temp, buffer_size=50)

        ref_dx, ref_dxdps = ref_intg.step_op()
        test_dx, test_dxdps = test_intg.step_op()

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        x_ref = np.copy(x0)
        x_test = np.copy(x0)
        for step in range(num_steps):

            ref_dx_val, ref_dxdp_val = sess.run([ref_dx, ref_dxdps], feed_dict={x_ph: x_ref})
            test_dx_val, test_dxdp_val = sess.run([test_dx, test_dxdps], feed_dict={x_ph: x_test})

            np.testing.assert_array_almost_equal(ref_dx_val, test_dx_val, decimal=14)
            np.testing.assert_array_almost_equal(ref_dxdp_val, test_dxdp_val, decimal=14) # BAD WTF CONVERGENCE

            x_ref += ref_dx_val
            x_test += test_dx_val


    def test_ten_steps_with_periodic_box(self):
        """
        Testing that we can integrate with a periodic box.
        """
        friction = 10.0
        dt = 0.003
        temp = 0.0

        masses = np.array([1.0, 12.0, 4.0, 2.0, 3.0])
        num_atoms = len(masses)
        x_ph = tf.placeholder(dtype=tf.float64, shape=(num_atoms, 3))

        x0 = np.array([
            [ 0.0637,   0.0126,   0.2203],
            [ 1.0573,  -0.2011,   1.2864],
            [ 2.3928,   1.2209,  -0.2230],
            [-0.6891,   1.6983,   0.0780],
            [-0.6312,  -1.6261,  -0.2601]
        ], dtype=np.float64)
        x0.setflags(write=False)

        b0 = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        b0.setflags(write=False)

        # one of the particles moves into infinity lol
        params = tf.convert_to_tensor(np.array([0.2, -0.2], dtype=np.float64))
        param_idxs = np.array([0, 1, 1, 1, 1], dtype=np.int32)

        exclusions = np.array([
            [0,0,1,0,0],
            [0,0,0,1,1],
            [1,0,0,0,0],
            [0,1,0,0,1],
            [0,1,0,1,0],
            ], dtype=np.bool)

        box_ph = tf.placeholder(shape=(3), dtype=tf.float64)
        electrostatic = Electrostatic(params, param_idxs, exclusions, box_ph)

        sess = tf.Session()
        
        ref_intg = ReferenceLangevinIntegrator(masses, dt, friction, temp)

        num_steps = 5

        x = x_ph
        box = box_ph

        for step in range(num_steps):
            print("step", step)
            all_grads = []
            all_box_grads = []
            for nrg in [electrostatic]:
                nrg_op = nrg.energy(x, box)
                all_grads.append(tf.gradients(nrg_op, x)[0])
                all_box_grads.append(tf.gradients(nrg_op, box)[0])

            grads = tf.reduce_sum(tf.stack(all_grads, axis=0), axis=0)
            box_grads = tf.reduce_sum(tf.stack(all_box_grads, axis=0), axis=0)
            
            dx = ref_intg.step(grads)
            x += dx
            box -= dt*box_grads

        ref_x_final_op = x
        ref_box_final_op = box

        # x = [5,3], params = [2], res = [5, 3, 2], but we typically expect [2,5,3]
        ref_dxdp_op = jacobian(x, electrostatic.get_params(), use_pfor=False)

        a, b, c = sess.run(
            [ref_x_final_op, ref_box_final_op, ref_dxdp_op],
            feed_dict={x_ph: x0, box_ph: b0}
        )

        test_intg = integrator.LangevinIntegrator(masses, x_ph, box_ph, [electrostatic], dt, friction, temp)
        [dx_op, dxdps_op], [db_op, dbdps_op] = test_intg.step_op()

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        # ref_x_final, ref_dxdp_hb, ref_dxdp_ha = sess.run([ref_x_final_op, ref_dxdp_hb_op, ref_dxdp_ha_op], feed_dict={x_ph: self.x0})

        x = np.copy(x0) # this copy is super important else it just modifies everything in place
        b = np.copy(b0)
        for step in range(num_steps):
            # dxdp_ops returns [5,3,2]
            dx_val, dxdp_val, db_val, dbdp_val = sess.run([dx_op, dxdps_op, db_op, dbdps_op], feed_dict={x_ph: x, box_ph: b})
            x += dx_val
            b += db_val


        test_dxdp = dxdp_val
        test_x_final_val = x

        np.testing.assert_array_almost_equal(test_x_final_val, a, decimal=13)
        np.testing.assert_array_almost_equal(dxdp_val, np.transpose(c, (2,0,1)), decimal=13) # PASSED WTf


    def test_ten_steps(self):
        """
        Testing against reference implementation.
        """
        masses = np.array([1.0, 12.0, 4.0])
        x0 = np.array([
            [1.0, 0.5, -0.5],
            [0.2, 0.1, -0.3],
            [0.5, 0.4, 0.3],
        ], dtype=np.float64)
        x0.setflags(write=False)

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

        hb = bonded_force.HarmonicBond(
            bond_params,
            bond_idxs,
            param_idxs,
        )

        angle_params = [
            tf.get_variable("HCH_ka", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(np.sqrt(75.0))),
            tf.get_variable("HCH_a0", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(1.81)),
        ]

        ha = bonded_force.HarmonicAngle(
            params=angle_params,
            angle_idxs=np.array([[1,0,2]], dtype=np.int32),
            param_idxs=np.array([[0,1]], dtype=np.int32)
        )

        friction = 10.0
        dt = 0.003
        temp = 0.0
        num_atoms = len(masses)
        x_ph = tf.placeholder(dtype=tf.float64, shape=(num_atoms, 3))


        ref_intg = ReferenceLangevinIntegrator(masses, dt, friction, temp)

        num_steps = 5

        x = x_ph

        for step in range(num_steps):
            print("step", step)
            all_grads = []
            for nrg in [hb, ha]:
                all_grads.append(tf.gradients(nrg.energy(x), x)[0])
            all_grads = tf.stack(all_grads, axis=0)
            grads = tf.reduce_sum(all_grads, axis=0)
            dx = ref_intg.step(grads)
            x += dx

        ref_x_final_op = x

        # verify correctness of jacobians through time
        ref_dxdp_hb_op = jacobian(x, hb.get_params(), use_pfor=False)
        ref_dxdp_ha_op = jacobian(x, ha.get_params(), use_pfor=False)

        test_intg = integrator.LangevinIntegrator(masses, x_ph, None, [hb, ha], dt, friction, temp)
        dx_op, dxdps_op = test_intg.step_op()
        # dxdps_op = tf.reduce_sum(dxdps_op, axis=[1,2])

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        ref_x_final, ref_dxdp_hb, ref_dxdp_ha = sess.run([ref_x_final_op, ref_dxdp_hb_op, ref_dxdp_ha_op], feed_dict={x_ph: x0})

        x = np.copy(x0) # this copy is super important else it just modifies everything in place
        for step in range(num_steps):
            [dx_val, dxdp_val] = sess.run([dx_op, dxdps_op], feed_dict={x_ph: x})
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

        ref_dLdp_op = tf.gradients(l0, hb.params+ha.params) # goes through reference integrator
        test_dLdx_op = tf.gradients(l1, x_final_ph)
        test_dLdp_op_gvs = test_intg.grads_and_vars(test_dLdx_op[0]) # multiply with dxdp

        # need to fix this test. 
        ref_dLdp = sess.run(ref_dLdp_op, feed_dict={x_ph: x0})
        test_dLdp = sess.run([a[0] for a in test_dLdp_op_gvs], feed_dict={x_final_ph: test_x_final_val})

        np.testing.assert_array_almost_equal(ref_dLdp, test_dLdp)


if __name__ == "__main__":

    unittest.main()
