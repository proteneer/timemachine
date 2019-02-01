import unittest
import numpy as np
import tensorflow as tf
from timemachine.functionals import bonded
from tensorflow.python.ops.parallel_for.gradients import jacobian
from timemachine import derivatives

class PeriodicTorsionForceOpenMM():

    def __init__(self,
        params,
        torsion_idxs,
        param_idxs,
        precision=tf.float64):
        """
        Implements the OpenMM version of PeriodicTorsionForce 
        """
        self.params = params
        self.torsion_idxs = torsion_idxs
        self.param_idxs = param_idxs

    @staticmethod
    def get_signed_angle(ci, cj, ck, cl):
        rij = ci - cj
        rkj = ck - cj
        rkl = ck - cl

        n1 = tf.cross(rij, rkj)
        n2 = tf.cross(rkj, rkl)

        lhs = tf.norm(n1, axis=-1)
        rhs = tf.norm(n2, axis=-1)
        bot = lhs * rhs

        top = tf.reduce_sum(tf.multiply(n1, n2), -1)
        cos_angles = top/bot # debug

        sign = tf.sign(tf.reduce_sum(tf.multiply(rkj, tf.cross(n1, n2)), -1))
        angle = sign * tf.acos(cos_angles)

        return angle

    def energy(self, conf):
        ci = tf.gather(conf, self.torsion_idxs[:, 0])
        cj = tf.gather(conf, self.torsion_idxs[:, 1])
        ck = tf.gather(conf, self.torsion_idxs[:, 2])
        cl = tf.gather(conf, self.torsion_idxs[:, 3])

        ks = tf.gather(self.params, self.param_idxs[:, 0])
        phase = tf.gather(self.params, self.param_idxs[:, 1])
        period = tf.gather(self.params, self.param_idxs[:, 2])

        angle = self.get_signed_angle(ci, cj, ck, cl) # should have shape (4,)
        # angle = tf.reduce_sum(ci+cj+ck+cl)

        e0 = ks*(1+tf.cos(period * angle - phase)) # cos(a) cos(t0) + sin(a) sin(t0)
        return tf.reduce_sum(e0, axis=-1)

class TestBonded(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_harmonic_bond(self):
        x0 = np.array([
            [1.0, 0.2, 3.3], # H 
            [-0.5,-1.1,-0.9], # C
        ], dtype=np.float64)

        bond_params = [
            tf.get_variable("HC_kb", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(10.0)),
            tf.get_variable("HC_b0", shape=tuple(), dtype=tf.float64, initializer=tf.constant_initializer(3.0))
        ]

        hb = bonded.HarmonicBond(
            params=bond_params,
            bond_idxs=np.array([[0,1]], dtype=np.int32),
            param_idxs=np.array([[0,1]], dtype=np.int32)
        )

        x_ph = tf.placeholder(shape=(2,3), dtype=np.float64)

        nrg_op = hb.energy(x_ph)

        hess_op = tf.hessians(nrg_op, x_ph)


        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        jac_op = jacobian(derivatives.densify(tf.gradients(nrg_op, x_ph)[0]), bond_params, use_pfor=False)

        print(sess.run(hess_op, feed_dict={x_ph: x0})[0].reshape((2, -1)))

        print(sess.run(jac_op, feed_dict={x_ph: x0})[0].reshape((2, -1)))
        print(sess.run(jac_op, feed_dict={x_ph: x0})[1].reshape((2, -1)))




class TestBondedForce(unittest.TestCase):

    def setUp(self):
        self.conformers = np.array([
            [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
             [ 0.561317027011325 , 0.2066950040043141, 0.3670430960815993],
             [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
             [ 0.9399773448903637,-0.6888774474110431, 0.2104211949995816]],
            [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
             [ 0.5613170270113252, 0.2066950040043142, 0.3670430960815993],
             [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
             [ 1.283345455745044 ,-0.0356257425880843,-0.2573923896494185]],
            [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
             [ 0.561317027011325 , 0.2066950040043142, 0.3670430960815992],
             [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
             [ 1.263820400176392 , 0.7964992122869241, 0.0084568741589791]],
            [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
             [ 0.5613170270113252, 0.2066950040043142, 0.3670430960815992],
             [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
             [ 0.8993534242298198, 1.042445571242743 , 0.7635483993060286]],
            [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
             [ 0.5613170270113255, 0.2066950040043142, 0.3670430960815993],
             [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
             [ 0.5250337847650304, 0.476091386095139 , 1.3136545198545133]],
            [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
             [ 0.5613170270113255, 0.2066950040043141, 0.3670430960815993],
             [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
             [ 0.485009232042489 ,-0.3818599172073237, 1.1530102055165103]],
            ], dtype=np.float64)

        self.nan_conformers = np.array([
            [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
             [ 0.5613170270113252, 0.2066950040043142, 0.3670430960815993],
             [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
             [ 1.2278668040866427, 0.8805184219394547, 0.099391329616366 ]],
            [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
             [ 0.561317027011325 , 0.206695004004314 , 0.3670430960815994],
             [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
             [ 0.5494071252089705,-0.5626592973923106, 0.9817919758125693]],
            ], dtype=np.float64)

    def tearDown(self):
        tf.reset_default_graph()

    def test_torsions_with_openmm(self):
        """
        Test agreement of torsions with OpenMM's implementation of torsion terms.
        """

        torsion_idxs = np.array([
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ], dtype=np.int32)

        params = tf.get_variable(name="parameters", dtype=tf.float64, shape=(9,), initializer=tf.constant_initializer([
            2.3, # k0
            5.4, # k1
            9.0, # k2
            0.0, # t0
            3.0, # t1
            5.8, # t2
            1.0, # n0
            2.0, # n1
            3.0  # n2
        ]))

        param_idxs = np.array([
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8]
        ], dtype=np.int32)

        x_ph = tf.placeholder(shape=(4, 3), dtype=tf.float64)

        test_force = bonded.PeriodicTorsion(params, torsion_idxs, param_idxs)
        ref_force = PeriodicTorsionForceOpenMM(params, torsion_idxs, param_idxs)

        test_nrg = test_force.energy(x_ph)
        test_grad, test_hessian, test_mixed = derivatives.compute_ghm(test_nrg, x_ph, [params])

        ref_nrg = ref_force.energy(x_ph)
        ref_grad, ref_hessian, ref_mixed = derivatives.compute_ghm(ref_nrg, x_ph, [params])

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        for conf_idx, conf in enumerate(self.conformers):

            t0, t1, t2, t3, r0, r1, r2, r3 = sess.run([
                test_nrg,
                derivatives.densify(test_grad),
                test_hessian,
                test_mixed,
                ref_nrg,
                derivatives.densify(ref_grad),
                ref_hessian,
                ref_mixed], feed_dict={x_ph: conf})

            np.testing.assert_array_almost_equal(t0, r0, decimal=14) # energy
            np.testing.assert_array_almost_equal(t1, r1, decimal=13) # grad
            np.testing.assert_array_almost_equal(t2, r2, decimal=12) # hessian
            np.testing.assert_array_almost_equal(t3, r3, decimal=13) # mixed partials

            # net force should be zero
            np.testing.assert_almost_equal(np.sum(t1, axis=0), [0,0,0], decimal=14)

        # OpenMM's vanilla implementation of the energy is non-differentiable when
        # the angle is equal to zero. The atan2 version is numerically stable since
        # it avoids taking a derivative of an arccos.
        for conf_idx, conf in enumerate(self.nan_conformers):

            t0, t1, t2, t3, r0, r1, r2, r3 = sess.run([
                test_nrg,
                derivatives.densify(test_grad),
                test_hessian,
                test_mixed,
                ref_nrg,
                derivatives.densify(ref_grad),
                ref_hessian,
                ref_mixed], feed_dict={x_ph: conf})

            np.testing.assert_array_almost_equal(t0, r0, decimal=14) # energy

            assert not np.any(np.isnan(t1))
            # assert np.any(np.isnan(r1))
            assert not np.any(np.isnan(t2))
            # assert np.any(np.isnan(r2))
            assert not np.any(np.isnan(t3))
            # assert np.any(np.isnan(r3))

            # net force should be zero
            np.testing.assert_almost_equal(np.sum(t1, axis=0), [0,0,0], decimal=14)



if __name__ == "__main__":
    unittest.main()


