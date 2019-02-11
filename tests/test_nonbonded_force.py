import unittest
import numpy as np
import tensorflow as tf
from timemachine.functionals.nonbonded import Electrostatic, LeonnardJones
from timemachine.cpu_functionals import custom_ops
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

# class TestLennardJones(unittest.TestCase):

#     def tearDown(self):
#         tf.reset_default_graph()

#     def test_lj612_cpu(self):

#         x0 = np.array([
#             [ 0.0637,   0.0126,   0.2203],
#             [ 1.0573,  -0.2011,   1.2864],
#             [ 2.3928,   1.2209,  -0.2230],
#             [-0.6891,   1.6983,   0.0780],
#             [-0.6312,  -1.6261,  -0.2601]
#         ], dtype=np.float64)

#         x_ph = tf.placeholder(shape=(5, 3), dtype=np.float64)

#         params_np = np.array([3.0, 2.0, 1.0, 1.4], dtype=np.float64)
#         params_tf = tf.convert_to_tensor(params_np)
#         param_idxs = np.array([
#             [0, 3],
#             [1, 2],
#             [1, 2],
#             [1, 2],
#             [1, 2]], dtype=np.int32)

#         scale_matrix = np.array([
#             [  0,  0,  1,0.5,  0],
#             [  0,  0,  0,  1,  1],
#             [  1,  0,  0,  0,0.2],
#             [0.5,  1,  0,  0,  1],
#             [  0,  1,0.2,  1,  0],
#         ], dtype=np.float64)

#         sess = tf.Session()
#         sess.run(tf.initializers.global_variables())

#         cutoff = None

#         ref_nrg = LeonnardJones(params_tf, param_idxs, scale_matrix, cutoff=cutoff)
#         nrg_op = ref_nrg.energy(x_ph)

#         test_lj = custom_ops.LennardJones_double(
#             params_np.reshape(-1).tolist(),
#             list(range(params_np.shape[0])),
#             param_idxs.reshape(-1).tolist(),
#             scale_matrix.reshape(-1).tolist()
#         )


#         ref_grad, ref_hessians, ref_mps = derivatives.compute_ghm(nrg_op, x_ph, [params_tf])
#         test_nrg, test_grads, test_hessians, test_mps = test_lj.total_derivative(x0, params_np.shape[0])

#         sess = tf.Session()
#         np.testing.assert_array_almost_equal(test_nrg, sess.run(nrg_op, feed_dict={x_ph: x0}), decimal=13)
#         np.testing.assert_array_almost_equal(test_grads, sess.run(ref_grad, feed_dict={x_ph: x0}), decimal=12)
#         # tighten the tolerance for this later.
#         np.testing.assert_array_almost_equal(test_hessians, sess.run(ref_hessians, feed_dict={x_ph: x0}), decimal=11)
#         np.testing.assert_array_almost_equal(test_mps, sess.run(ref_mps[0], feed_dict={x_ph: x0}), decimal=12)


#     def test_lj612(self):
#         """
#         Testing non-periodic LJ 612 forces.
#         """
#         x0 = np.array([
#             [ 0.0637,   0.0126,   0.2203],
#             [ 1.0573,  -0.2011,   1.2864],
#             [ 2.3928,   1.2209,  -0.2230],
#             [-0.6891,   1.6983,   0.0780],
#             [-0.6312,  -1.6261,  -0.2601]
#         ], dtype=np.float64)

#         x_ph = tf.placeholder(shape=(5, 3), dtype=np.float64)

#         params_np = np.array([3.0, 2.0, 1.0, 1.4], dtype=np.float64)
#         params = tf.convert_to_tensor(params_np)
#         # Ai, Ci
#         param_idxs = np.array([
#             [0, 3],
#             [1, 2],
#             [1, 2],
#             [1, 2],
#             [1, 2]], dtype=np.int32)

#         scale_matrix = np.array([
#             [  0,  0,  1,0.5,  0],
#             [  0,  0,  0,  1,  1],
#             [  1,  0,  0,  0,0.2],
#             [0.5,  1,  0,  0,  1],
#             [  0,  1,0.2,  1,  0],
#         ], dtype=np.float64)

#         # Test 1: no cutoffs
#         lj = LeonnardJones(params, param_idxs, scale_matrix, cutoff=None)

#         sess = tf.Session()
#         sess.run(tf.initializers.global_variables())

#         num_atoms = x0.shape[0]

#         rlj = ReferenceLJEnergy(params_np, param_idxs, scale_matrix, cutoff=None)
#         ref_nrg = rlj.energy(x0)

#         # (ytz) TODO: add a test for forces, for now we rely on
#         # autograd to be analytically correct.
#         test_nrg_op = lj.energy(x_ph)
#         test_nrg = sess.run(test_nrg_op, feed_dict={x_ph: x0})
#         np.testing.assert_almost_equal(ref_nrg, test_nrg, decimal=8)

#         test_grads_op = tf.gradients(test_nrg_op, x_ph)[0]
#         test_grads = sess.run(test_grads_op, feed_dict={x_ph: x0})

#         net_force = np.sum(test_grads, axis=0)
#         # this also checks that there are no NaNs

#         np.testing.assert_almost_equal(net_force, [0, 0, 0], decimal=7)

#         assert not np.any(np.isnan(sess.run(tf.hessians(test_nrg_op, x_ph), feed_dict={x_ph: x0})))
#         mixed_partials = sess.run(derivatives.list_jacobian(test_grads_op, [lj.get_params()]), feed_dict={x_ph: x0})
#         assert not np.any(np.isnan(mixed_partials))

#         # Test 2: with cutoffs
#         lj = LeonnardJones(params, param_idxs, scale_matrix, cutoff=2.0)
#         sess = tf.Session()
#         sess.run(tf.initializers.global_variables())
#         num_atoms = x0.shape[0]
#         rlj = ReferenceLJEnergy(params_np, param_idxs, scale_matrix, cutoff=2.0)
#         ref_nrg = rlj.energy(x0)

#         # (ytz) TODO: add a test for forces, for now we rely on
#         # autograd to be analytically correct.
#         test_nrg_op = lj.energy(x_ph)
#         test_nrg = sess.run(test_nrg_op, feed_dict={x_ph: x0})
#         np.testing.assert_almost_equal(ref_nrg, test_nrg, decimal=8)
#         test_grads_op = tf.gradients(test_nrg_op, x_ph)[0]
#         test_grads = sess.run(test_grads_op, feed_dict={x_ph: x0})
#         net_force = np.sum(test_grads, axis=0)
#         # this also checks that there are no NaNs
#         np.testing.assert_almost_equal(net_force, [0, 0, 0], decimal=7)

#         assert not np.any(np.isnan(sess.run(tf.hessians(test_nrg_op, x_ph), feed_dict={x_ph: x0})))
#         mixed_partials = sess.run(derivatives.list_jacobian(test_grads_op, [lj.get_params()]), feed_dict={x_ph: x0})
#         assert not np.any(np.isnan(mixed_partials))


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

    def test_electrostatics_large(self):
        np.random.seed(0)
        x0 = np.random.rand(5, 3).astype(np.float64)
        print(x0)

        N = x0.shape[0]

        x_ph = tf.placeholder(shape=(N, 3), dtype=np.float64)


        num_params = 25

        params_np = np.random.rand(num_params).astype(np.float64)
        params_tf = tf.convert_to_tensor(params_np)
        param_idxs = np.random.randint(num_params, size=(N,), dtype=np.int32)
        scale_matrix = np.random.rand(N, N).astype(np.float64)
        scale_matrix = (scale_matrix + scale_matrix.T)/2
        np.fill_diagonal(scale_matrix, 0.0)

        cutoff = None
        crf = 0.0

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        ref_nrg = Electrostatic(params_tf, param_idxs, scale_matrix, cutoff=cutoff, crf=crf)
        nrg_op = ref_nrg.energy(x_ph)

        test_es = custom_ops.ElectrostaticsGPU_double(
            params_np.reshape(-1).tolist(),
            list(range(params_np.shape[0])),
            param_idxs.reshape(-1).tolist(),
            scale_matrix.reshape(-1).tolist()
        )

        ref_grad, ref_hessians, ref_mps = derivatives.compute_ghm(nrg_op, x_ph, [params_tf])
        test_nrg, test_grads, test_hessians, test_mps = test_es.total_derivative(x0, params_np.shape[0])

        sess = tf.Session()
        np.testing.assert_allclose(test_grads, sess.run(ref_grad, feed_dict={x_ph: x0}), rtol=1e-11)

        ref_h_val = sess.run(ref_hessians, feed_dict={x_ph: x0}).reshape(N*3, N*3)



        test_h_val = test_hessians.reshape(N*3, N*3)

        print(np.tril(ref_h_val), np.tril(test_h_val))
        np.testing.assert_allclose(np.tril(ref_h_val), np.tril(test_h_val), rtol=1e-11)


    # def test_electrostatics_gpu(self):
    #     x0 = np.array([
    #         [ 0.0637,   0.0126,   0.2203],
    #         [ 1.0573,  -0.2011,   1.2864],
    #         [ 2.3928,   1.2209,  -0.2230],
    #         [-0.6891,   1.6983,   0.0780],
    #         [-0.6312,  -1.6261,  -0.2601]
    #     ], dtype=np.float64)

    #     N = x0.shape[0]

    #     x_ph = tf.placeholder(shape=(N, 3), dtype=np.float64)

    #     params_np = np.array([1.3, 0.3], dtype=np.float64)
    #     params_tf = tf.convert_to_tensor(params_np)
    #     param_idxs = np.array([0, 1, 1, 1, 1], dtype=np.int32)
    #     scale_matrix = np.array([
    #         [  0,  1,  1,  1,0.5],
    #         [  1,  0,  0,  1,  1],
    #         [  1,  0,  0,  0,0.2],
    #         [  1,  1,  0,  0,  1],
    #         [0.5,  1,0.2,  1,  0],
    #     ], dtype=np.float64)

    #     cutoff = None
    #     crf = 0.0

    #     sess = tf.Session()
    #     sess.run(tf.initializers.global_variables())

    #     ref_nrg = Electrostatic(params_tf, param_idxs, scale_matrix, cutoff=cutoff, crf=crf)
    #     nrg_op = ref_nrg.energy(x_ph)

    #     test_es = custom_ops.ElectrostaticsGPU_double(
    #         params_np.reshape(-1).tolist(),
    #         list(range(params_np.shape[0])),
    #         param_idxs.reshape(-1).tolist(),
    #         scale_matrix.reshape(-1).tolist()
    #     )

    #     ref_grad, ref_hessians, ref_mps = derivatives.compute_ghm(nrg_op, x_ph, [params_tf])
    #     test_nrg, test_grads, test_hessians, test_mps = test_es.total_derivative(x0, params_np.shape[0])

    #     sess = tf.Session()
    #     np.testing.assert_array_almost_equal(test_grads, sess.run(ref_grad, feed_dict={x_ph: x0}), decimal=13)
    #     np.testing.assert_array_almost_equal(test_hessians, sess.run(ref_hessians, feed_dict={x_ph: x0}), decimal=11)
        # np.testing.assert_array_almost_equal(test_nrg, sess.run(nrg_op, feed_dict={x_ph: x0}), decimal=13)
        # np.testing.assert_array_almost_equal(test_mps, sess.run(ref_mps[0], feed_dict={x_ph: x0}), decimal=13)


    # def test_electrostatics_cpu(self):
    #     x0 = np.array([
    #         [ 0.0637,   0.0126,   0.2203],
    #         [ 1.0573,  -0.2011,   1.2864],
    #         [ 2.3928,   1.2209,  -0.2230],
    #         [-0.6891,   1.6983,   0.0780],
    #         [-0.6312,  -1.6261,  -0.2601]
    #     ], dtype=np.float64)

    #     x_ph = tf.placeholder(shape=(5, 3), dtype=np.float64)

    #     params_np = np.array([1.3, 0.3], dtype=np.float64)
    #     params_tf = tf.convert_to_tensor(params_np)
    #     param_idxs = np.array([0, 1, 1, 1, 1], dtype=np.int32)
    #     scale_matrix = np.array([
    #         [  0,  1,  1,  1,0.5],
    #         [  1,  0,  0,  1,  1],
    #         [  1,  0,  0,  0,0.2],
    #         [  1,  1,  0,  0,  1],
    #         [0.5,  1,0.2,  1,  0],
    #     ], dtype=np.float64)

    #     cutoff = None
    #     crf = 0.0

    #     sess = tf.Session()
    #     sess.run(tf.initializers.global_variables())

    #     ref_nrg = Electrostatic(params_tf, param_idxs, scale_matrix, cutoff=cutoff, crf=crf)
    #     nrg_op = ref_nrg.energy(x_ph)

    #     test_es = custom_ops.Electrostatics_double(
    #         params_np.reshape(-1).tolist(),
    #         list(range(params_np.shape[0])),
    #         param_idxs.reshape(-1).tolist(),
    #         scale_matrix.reshape(-1).tolist()
    #     )

    #     ref_grad, ref_hessians, ref_mps = derivatives.compute_ghm(nrg_op, x_ph, [params_tf])
    #     test_nrg, test_grads, test_hessians, test_mps = test_es.total_derivative(x0, params_np.shape[0])

    #     sess = tf.Session()
    #     np.testing.assert_array_almost_equal(test_nrg, sess.run(nrg_op, feed_dict={x_ph: x0}), decimal=13)
    #     np.testing.assert_array_almost_equal(test_grads, sess.run(ref_grad, feed_dict={x_ph: x0}), decimal=13)
    #     np.testing.assert_array_almost_equal(test_hessians, sess.run(ref_hessians, feed_dict={x_ph: x0}), decimal=12)
    #     np.testing.assert_array_almost_equal(test_mps, sess.run(ref_mps[0], feed_dict={x_ph: x0}), decimal=13)

    # def test_electrostatics(self):
    #     """
    #     Testing non-periodic electrostatic forces.
    #     """
    #     x0 = np.array([
    #         [ 0.0637,   0.0126,   0.2203],
    #         [ 1.0573,  -0.2011,   1.2864],
    #         [ 2.3928,   1.2209,  -0.2230],
    #         [-0.6891,   1.6983,   0.0780],
    #         [-0.6312,  -1.6261,  -0.2601]
    #     ], dtype=np.float64)

    #     x_ph = tf.placeholder(shape=(5, 3), dtype=np.float64)

    #     params_np = np.array([1.3, 0.3], dtype=np.float64)
    #     params = tf.convert_to_tensor(params_np)
    #     param_idxs = np.array([0, 1, 1, 1, 1], dtype=np.int32)
    #     scale_matrix = np.array([
    #         [  0,  1,  1,  1,0.5],
    #         [  1,  0,  0,  1,  1],
    #         [  1,  0,  0,  0,0.2],
    #         [  1,  1,  0,  0,  1],
    #         [0.5,  1,0.2,  1,  0],
    #     ], dtype=np.float64)


    #     cutoffs = [None, 2.0]
    #     crfs = [0.0, 1.0]

    #     # No cutoff, crf=1.0
    #     for cutoff in cutoffs:
    #         for crf in crfs:

    #             ef = Electrostatic(params, param_idxs, scale_matrix, cutoff=cutoff, crf=crf)

    #             sess = tf.Session()
    #             sess.run(tf.initializers.global_variables())

    #             reference = ReferenceElectrostatics(params_np, param_idxs, scale_matrix, cutoff=cutoff, crf=crf)
    #             ref_nrg = reference.energy(x0)

    #             # (ytz) TODO: add a test for forces, for now we rely on
    #             # autograd to be analytically correct.
    #             test_nrg_op = ef.energy(x_ph)
    #             test_nrg = sess.run(test_nrg_op, feed_dict={x_ph: x0})
    #             np.testing.assert_almost_equal(ref_nrg, test_nrg, decimal=13)

    #             test_grads_op = tf.gradients(test_nrg_op, x_ph)[0]
    #             test_grads = sess.run(test_grads_op, feed_dict={x_ph: x0})

    #             net_force = np.sum(test_grads, axis=0)
    #             # this also checks that there are no NaNs
    #             np.testing.assert_almost_equal(net_force, [0,0,0], decimal=14)

    #             assert not np.any(np.isnan(sess.run(tf.hessians(test_nrg_op, x_ph), feed_dict={x_ph: x0})))
    #             mixed_partials = sess.run(derivatives.list_jacobian(test_grads_op, [ef.get_params()]), feed_dict={x_ph: x0})
    #             assert not np.any(np.isnan(mixed_partials))


if __name__ == "__main__":
    unittest.main()