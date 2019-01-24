import unittest
import numpy as np
import tensorflow as tf
from timemachine.functionals.nonbonded import Electrostatic, LeonnardJones
from timemachine.constants import ONE_4PI_EPS0
from timemachine import derivatives



class ReferenceLJEnergy():

    def __init__(self, params, param_idxs, scale_matrix, cutoff=None):
        self.params = params
        self.param_idxs = param_idxs
        self.scale_matrix = scale_matrix
        self.cutoff = cutoff

    def energy(self, conf):
        ref_nrg = 0
        num_atoms = conf.shape[0]
        for i in range(num_atoms):
            sig_i = self.params[self.param_idxs[i, 0]]
            eps_i = self.params[self.param_idxs[i, 1]]
            ri = conf[i]

            for j in range(i+1, num_atoms):
                sig_j = self.params[self.param_idxs[j, 0]]
                eps_j = self.params[self.param_idxs[j, 1]]
                rj = conf[j]
                r = np.linalg.norm(conf[i] - conf[j])
                if self.cutoff is not None and r > self.cutoff:
                    continue
                sig = (sig_i + sig_j)/2
                sig2 = sig/r
                sig2 *= sig2
                sig6 = sig2*sig2*sig2
                eps = self.scale_matrix[i, j]*np.sqrt(eps_i * eps_j)
                vdwEnergy = 4*eps*(sig6-1.0)*sig6
                ref_nrg += vdwEnergy

        return ref_nrg

class TestLeonnardJones(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_lj612(self):
        """
        Testing non-periodic LJ 612 forces.
        """
        x0 = np.array([
            [ 0.0637,   0.0126,   0.2203],
            [ 1.0573,  -0.2011,   1.2864],
            [ 2.3928,   1.2209,  -0.2230],
            [-0.6891,   1.6983,   0.0780],
            [-0.6312,  -1.6261,  -0.2601]
        ], dtype=np.float64)

        x_ph = tf.placeholder(shape=(5, 3), dtype=np.float64)

        params_np = np.array([3.0, 2.0, 1.0, 1.4], dtype=np.float64)
        params = tf.convert_to_tensor(params_np)
        # Ai, Ci
        param_idxs = np.array([
            [0, 3],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2]], dtype=np.int32)

        scale_matrix = np.array([
            [  0,  0,  1,0.5,  0],
            [  0,  0,  0,  1,  1],
            [  1,  0,  0,  0,0.2],
            [0.5,  1,  0,  0,  1],
            [  0,  1,0.2,  1,  0],
        ], dtype=np.float64)

        # Test 1: no cutoffs
        lj = LeonnardJones(params, param_idxs, scale_matrix, cutoff=None)

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        num_atoms = x0.shape[0]

        rlj = ReferenceLJEnergy(params_np, param_idxs, scale_matrix, cutoff=None)
        ref_nrg = rlj.energy(x0)

        # (ytz) TODO: add a test for forces, for now we rely on
        # autograd to be analytically correct.
        test_nrg_op = lj.energy(x_ph)
        test_nrg = sess.run(test_nrg_op, feed_dict={x_ph: x0})
        np.testing.assert_almost_equal(ref_nrg, test_nrg, decimal=8)

        test_grads_op = tf.gradients(test_nrg_op, x_ph)[0]
        test_grads = sess.run(test_grads_op, feed_dict={x_ph: x0})

        net_force = np.sum(test_grads, axis=0)
        # this also checks that there are no NaNs

        np.testing.assert_almost_equal(net_force, [0, 0, 0], decimal=7)

        assert not np.any(np.isnan(sess.run(tf.hessians(test_nrg_op, x_ph), feed_dict={x_ph: x0})))
        mixed_partials = sess.run(derivatives.list_jacobian(test_grads_op, [lj.get_params()]), feed_dict={x_ph: x0})
        assert not np.any(np.isnan(mixed_partials))

        # Test 2: with cutoffs
        lj = LeonnardJones(params, param_idxs, scale_matrix, cutoff=2.0)
        sess = tf.Session()
        sess.run(tf.initializers.global_variables())
        num_atoms = x0.shape[0]
        rlj = ReferenceLJEnergy(params_np, param_idxs, scale_matrix, cutoff=2.0)
        ref_nrg = rlj.energy(x0)

        # (ytz) TODO: add a test for forces, for now we rely on
        # autograd to be analytically correct.
        test_nrg_op = lj.energy(x_ph)
        test_nrg = sess.run(test_nrg_op, feed_dict={x_ph: x0})
        np.testing.assert_almost_equal(ref_nrg, test_nrg, decimal=8)
        test_grads_op = tf.gradients(test_nrg_op, x_ph)[0]
        test_grads = sess.run(test_grads_op, feed_dict={x_ph: x0})
        net_force = np.sum(test_grads, axis=0)
        # this also checks that there are no NaNs
        np.testing.assert_almost_equal(net_force, [0, 0, 0], decimal=7)

        assert not np.any(np.isnan(sess.run(tf.hessians(test_nrg_op, x_ph), feed_dict={x_ph: x0})))
        mixed_partials = sess.run(derivatives.list_jacobian(test_grads_op, [lj.get_params()]), feed_dict={x_ph: x0})
        assert not np.any(np.isnan(mixed_partials))


class ReferenceElectrostatics():

    def __init__(self, params, param_idxs, scale_matrix, cutoff=None, crf=1.0):
        self.params = params
        self.param_idxs = param_idxs
        self.scale_matrix = scale_matrix
        self.cutoff = cutoff
        self.crf = crf

    def energy(self, conf):
        ref_nrg = 0
        num_atoms = conf.shape[0]

        ref_nrg = 0
        for i in range(num_atoms):
            qi = self.params[self.param_idxs[i]]
            for j in range(i+1, num_atoms):
                dij = np.linalg.norm(conf[i] - conf[j])

                qj = self.params[self.param_idxs[j]]
                qij = qi * qj
                qij = self.scale_matrix[i, j] * qij

                if self.cutoff is None:
                    ref_nrg += qij/dij
                else:

                    if dij > self.cutoff:
                        continue
                    if self.scale_matrix[i, j] == 1.0:
                        dij_inverse = 1/dij - self.crf
                    else:
                        dij_inverse = 1/dij
                    ref_nrg += qij*dij_inverse

        return ONE_4PI_EPS0*ref_nrg

class TestElectrostatics(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_electrostatics(self):
        """
        Testing non-periodic electrostatic forces.
        """
        x0 = np.array([
            [ 0.0637,   0.0126,   0.2203],
            [ 1.0573,  -0.2011,   1.2864],
            [ 2.3928,   1.2209,  -0.2230],
            [-0.6891,   1.6983,   0.0780],
            [-0.6312,  -1.6261,  -0.2601]
        ], dtype=np.float64)

        x_ph = tf.placeholder(shape=(5, 3), dtype=np.float64)

        params_np = np.array([1.3, 0.3], dtype=np.float64)
        params = tf.convert_to_tensor(params_np)
        param_idxs = np.array([0, 1, 1, 1, 1], dtype=np.int32)
        scale_matrix = np.array([
            [  0,  1,  1,  1,0.5],
            [  1,  0,  0,  1,  1],
            [  1,  0,  0,  0,0.2],
            [  1,  1,  0,  0,  1],
            [0.5,  1,0.2,  1,  0],
        ], dtype=np.float64)


        cutoffs = [None, 2.0]
        crfs = [0.0, 1.0]

        # No cutoff, crf=1.0
        for cutoff in cutoffs:
            for crf in crfs:

                ef = Electrostatic(params, param_idxs, scale_matrix, cutoff=cutoff, crf=crf)

                sess = tf.Session()
                sess.run(tf.initializers.global_variables())

                reference = ReferenceElectrostatics(params_np, param_idxs, scale_matrix, cutoff=cutoff, crf=crf)
                ref_nrg = reference.energy(x0)

                # (ytz) TODO: add a test for forces, for now we rely on
                # autograd to be analytically correct.
                test_nrg_op = ef.energy(x_ph)
                test_nrg = sess.run(test_nrg_op, feed_dict={x_ph: x0})
                np.testing.assert_almost_equal(ref_nrg, test_nrg, decimal=13)

                test_grads_op = tf.gradients(test_nrg_op, x_ph)[0]
                test_grads = sess.run(test_grads_op, feed_dict={x_ph: x0})

                net_force = np.sum(test_grads, axis=0)
                # this also checks that there are no NaNs
                np.testing.assert_almost_equal(net_force, [0,0,0], decimal=14)

                assert not np.any(np.isnan(sess.run(tf.hessians(test_nrg_op, x_ph), feed_dict={x_ph: x0})))
                mixed_partials = sess.run(derivatives.list_jacobian(test_grads_op, [ef.get_params()]), feed_dict={x_ph: x0})
                assert not np.any(np.isnan(mixed_partials))


if __name__ == "__main__":
    unittest.main()