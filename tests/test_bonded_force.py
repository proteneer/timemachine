import unittest
import numpy as np
import tensorflow as tf
from timemachine.functionals import bonded
from tensorflow.python.ops.parallel_for.gradients import jacobian
from timemachine import derivatives
from timemachine.cpu_functionals import energy


class TestAngles(unittest.TestCase):

    def test_harmonic_angle(self):
        masses = np.array([6.0, 1.0, 1.0, 1.0, 1.0])
        x0 = np.array([
            [ 0.0637,   0.0126,   0.2203], # C
            [ 1.0573,  -0.2011,   1.2864], # H
            [ 2.3928,   1.2209,  -0.2230], # H
            [-0.6891,   1.6983,   0.0780], # H
            [-0.6312,  -1.6261,  -0.2601], # H
        ], dtype=np.float64)
        num_atoms = len(masses)
        angle_params_np = np.array([75, 1.91, 0.45], dtype=np.float64)
        angle_params_tf = tf.convert_to_tensor(angle_params_np)


        angle_idxs = np.array([[1,0,2],[1,0,3],[1,0,4],[2,0,3],[2,0,4],[3,0,4]])
        param_idxs = np.array([[0,1],[0,1],[0,2],[0,1],[0,1],[0,2]])

        ref_ha = bonded.HarmonicAngle(
            params=angle_params_tf,
            angle_idxs=angle_idxs,
            param_idxs=param_idxs,
            cos_angles=True
        )

        test_angle = energy.HarmonicAngle_double(
            angle_params_np.reshape(-1).tolist(),
            list(range(angle_params_np.shape[0])),
            param_idxs.reshape(-1).tolist(),
            angle_idxs.reshape(-1).tolist(),
            True,
        )

        x_ph = tf.placeholder(shape=x0.shape, dtype=np.float64)
        nrg_op = ref_ha.energy(x_ph)
        ref_grad, ref_hessians, ref_mps = derivatives.compute_ghm(nrg_op, x_ph, [angle_params_tf])
        test_nrg, test_grads, test_hessians, test_mps = test_angle.total_derivative(x0, angle_params_np.shape[0])

        sess = tf.Session()
        np.testing.assert_array_almost_equal(test_nrg, sess.run(nrg_op, feed_dict={x_ph: x0}), decimal=13)
        np.testing.assert_array_almost_equal(test_grads, sess.run(ref_grad, feed_dict={x_ph: x0}), decimal=13)
        np.testing.assert_array_almost_equal(test_hessians, sess.run(ref_hessians, feed_dict={x_ph: x0}), decimal=13)
        np.testing.assert_array_almost_equal(test_mps, sess.run(ref_mps[0], feed_dict={x_ph: x0}), decimal=13)


class TestBonded(unittest.TestCase):

    def test_harmonic_bond(self):
        x0 = np.array([
            [1.0, 0.2, 3.3], # H 
            [-0.5,-1.1,-0.9], # C
            [3.4, 5.5, 0.2], # H 
        ], dtype=np.float64)

        bond_params_np = np.array([10.0, 3.0, 5.5], dtype=np.float64)
        bond_params_tf = tf.convert_to_tensor(bond_params_np)
        param_idxs = np.array([
            [0,1],
            [1,2],
        ], dtype=np.int32)
        bond_idxs = np.array([[0,1], [1,2]], dtype=np.int32)

        ref_hb = bonded.HarmonicBond(
            params=bond_params_tf,
            param_idxs=param_idxs,
            bond_idxs=bond_idxs,
        )

        test_bond = energy.HarmonicBond_double(
            bond_params_np.reshape(-1).tolist(),
            list(range(bond_params_np.shape[0])),
            param_idxs.reshape(-1).tolist(),
            bond_idxs.reshape(-1).tolist(),
        )

        x_ph = tf.placeholder(shape=x0.shape, dtype=np.float64)
        nrg_op = ref_hb.energy(x_ph)
        ref_grad, ref_hessians, ref_mps = derivatives.compute_ghm(nrg_op, x_ph, [bond_params_tf])
        test_nrg, test_grads, test_hessians, test_mps = test_bond.total_derivative(x0, bond_params_np.shape[0])

        sess = tf.Session()
        np.testing.assert_array_almost_equal(test_nrg, sess.run(nrg_op, feed_dict={x_ph: x0}), decimal=13)
        np.testing.assert_array_almost_equal(test_grads, sess.run(ref_grad, feed_dict={x_ph: x0}), decimal=13)
        np.testing.assert_array_almost_equal(test_hessians, sess.run(ref_hessians, feed_dict={x_ph: x0}), decimal=13)
        np.testing.assert_array_almost_equal(test_mps, sess.run(ref_mps[0], feed_dict={x_ph: x0}), decimal=13)

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


class TestPeriodicTorsion(unittest.TestCase):

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

    def test_cpp_torsions(self):
        """
        Test agreement of torsions with OpenMM's implementation of torsion terms.
        """

        torsion_idxs = np.array([
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ], dtype=np.int32)

        params_np = np.array([
            2.3, # k0
            5.4, # k1
            9.0, # k2
            0.0, # t0
            3.0, # t1
            5.8, # t2
            1.0, # n0
            2.0, # n1
            3.0  # n2
        ])
        params_tf = tf.convert_to_tensor(params_np)

        param_idxs = np.array([
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8]
        ], dtype=np.int32)

        x_ph = tf.placeholder(shape=(4, 3), dtype=tf.float64)

        test_torsion = energy.PeriodicTorsion_double(
            params_np.reshape(-1),
            list(range(params_np.shape[0])),
            param_idxs.reshape(-1),
            torsion_idxs.reshape(-1)
        )
        ref_nrg = bonded.PeriodicTorsion(params=params_tf, param_idxs=param_idxs, torsion_idxs=torsion_idxs)

        for conf_idx, conf in enumerate(self.conformers):
            x_ph = tf.placeholder(shape=conf.shape, dtype=np.float64)
            nrg_op = ref_nrg.energy(x_ph)
            angle_op = ref_nrg.angles(x_ph)
            ref_grad, ref_hessians, ref_mps = derivatives.compute_ghm(nrg_op, x_ph, [params_tf])
            test_nrg, test_grads, test_hessians, test_mps = test_torsion.total_derivative(conf, params_np.shape[0])

            sess = tf.Session()
            np.testing.assert_array_almost_equal(test_nrg, sess.run(nrg_op, feed_dict={x_ph: conf}), decimal=13)
            np.testing.assert_array_almost_equal(test_grads, sess.run(ref_grad, feed_dict={x_ph: conf}), decimal=13)
            np.testing.assert_array_almost_equal(test_hessians, sess.run(ref_hessians, feed_dict={x_ph: conf}), decimal=13)
            np.testing.assert_array_almost_equal(test_mps, sess.run(ref_mps[0], feed_dict={x_ph: conf}), decimal=13)

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


