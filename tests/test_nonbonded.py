import unittest
import numpy as np
import functools

from jax.config import config; config.update("jax_enable_x64", True)
import jax
from jax.test_util import check_grads

from tests.invariances import assert_potential_invariance
from timemachine.potentials import nonbonded

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


class TestElectrostatics(unittest.TestCase):

    def test_periodic_electrostatics(self):
        conf = np.array([
            [ 0.0637,   0.0126,   0.2203],
            [ 1.0573,  -0.2011,   1.2864],
            [ 2.3928,   1.2209,  -0.2230],
            [-0.6891,   1.6983,   0.0780],
            [-0.6312,  -1.6261,  -0.2601]
        ], dtype=np.float64)

        params = np.array([1.3, 0.3], dtype=np.float64)
        param_idxs = np.array([0, 1, 1, 1, 1], dtype=np.int32)
        scale_matrix = np.array([
            [  0,  1,  1,  1,0.5],
            [  1,  0,  0,  1,  1],
            [  1,  0,  0,  0,0.2],
            [  1,  1,  0,  0,  1],
            [0.5,  1,0.2,  1,  0],
        ], dtype=np.float64)

        # warning: non net-neutral cell
        # ref_nrg = nonbonded.electrostatic(param_idxs, scale_matrix)

        box = np.array([
            [2.0, 0.0, 0.0],
            [0.6, 1.6, 0.0],
            [0.4, 0.7, 1.1]
        ], dtype=np.float64)

        energy_fn = functools.partial(
            nonbonded.electrostatics,
            param_idxs=param_idxs,
            scale_matrix=scale_matrix,
            cutoff=0.5,
            alpha=1.0,
            kmax=10)

        assert_potential_invariance(energy_fn, conf, params, box)

class TestLennardJones(unittest.TestCase):

    # def test_lj612_large(self):

    #     # THC
    #     x0 = np.array([
    #         [-3.7431,    0.0007,    3.3896],
    #         [-2.2513,    0.2400,    3.0656],
    #         [-1.8353,   -0.3114,    1.6756],
    #         [-0.3457,   -0.0141,    1.3424],
    #         [ 0.0913,   -0.5912,   -0.0304],
    #         [ 1.5230,   -0.3150,   -0.3496],
    #         [ 2.4800,   -1.3537,   -0.3510],
    #         [ 3.8242,   -1.1091,   -0.6908],
    #         [ 4.2541,    0.2018,   -1.0168],
    #         [ 3.2918,    1.2461,   -1.0393],
    #         [ 1.9507,    0.9887,   -0.6878],
    #         [ 3.6069,    2.5311,   -1.4305],
    #         [ 4.6870,    2.6952,   -2.3911],
    #         [ 5.9460,    1.9841,   -1.7744],
    #         [ 5.6603,    0.4771,   -1.4483],
    #         [ 6.7153,    0.0454,   -0.5274],
    #         [ 8.0153,    0.3238,   -0.7754],
    #         [ 8.3940,    1.0806,   -1.9842],
    #         [ 7.3027,    2.0609,   -2.5505],
    #         [ 9.0311,   -0.1319,    0.1662],
    #         [ 4.2434,    2.1598,   -3.7921],
    #         [ 4.9088,    4.2364,   -2.4878],
    #         [ 4.6917,   -2.1552,   -0.7266],
    #         [-3.9733,    0.4081,    4.3758],
    #         [-3.9690,   -1.0674,    3.3947],
    #         [-4.3790,    0.4951,    2.6522],
    #         [-1.6465,   -0.2405,    3.8389],
    #         [-2.0559,    1.3147,    3.1027],
    #         [-1.9990,   -1.3929,    1.6608],
    #         [-2.4698,    0.1371,    0.9054],
    #         [-0.1921,    1.0695,    1.3452],
    #         [ 0.2880,   -0.4452,    2.1239],
    #         [-0.0916,   -1.6686,   -0.0213],
    #         [-0.5348,   -0.1699,   -0.8201],
    #         [ 2.2004,   -2.3077,   -0.1063],
    #         [ 1.2776,    1.7607,   -0.6991],
    #         [ 6.1198,    2.5014,   -0.8189],
    #         [ 5.7881,   -0.1059,   -2.3685],
    #         [ 6.4691,   -0.4538,    0.2987],
    #         [ 9.3048,    1.6561,   -1.8023],
    #         [ 8.6369,    0.3417,   -2.7516],
    #         [ 7.6808,    3.0848,   -2.5117],
    #         [ 7.1355,    1.8275,   -3.6048],
    #         [ 8.8403,    0.2961,    1.1526],
    #         [10.0386,    0.1617,   -0.1353],
    #         [ 9.0076,   -1.2205,    0.2406],
    #         [ 4.1653,    1.0737,   -3.8000],
    #         [ 4.9494,    2.4548,   -4.5696],
    #         [ 3.2631,    2.5647,   -4.0515],
    #         [ 3.9915,    4.7339,   -2.8073],
    #         [ 5.1949,    4.6493,   -1.5175],
    #         [ 5.6935,    4.4750,   -3.2076],
    #         [ 4.1622,   -2.9559,   -0.5467]
    #     ], dtype=np.float64)

    #     N = x0.shape[0]

    #     x_ph = tf.placeholder(shape=(N, 3), dtype=np.float64)

    #     num_params = 4
    #     # params_np = (np.random.rand(num_params)+1.0).astype(dtype=np.float64)
    #     params_np = np.array([3.0, 2.0, 1.0, 1.4, 2.0, 2.0])
    #     params_tf = tf.convert_to_tensor(params_np)
    #     param_idxs = np.random.randint(num_params, size=(N,2))

    #     scale_matrix = np.random.rand(N, N).astype(np.float64)
    #     scale_matrix = (scale_matrix + scale_matrix.T)/2
    #     np.fill_diagonal(scale_matrix, 0.0)


    #     sess = tf.Session()
    #     sess.run(tf.initializers.global_variables())

    #     cutoff = None

    #     ref_nrg = LeonnardJones(params_tf, param_idxs, scale_matrix, cutoff=cutoff)
    #     nrg_op = ref_nrg.energy(x_ph)

    #     test_lj = custom_ops.LennardJonesGPU_double(
    #         params_np.reshape(-1).tolist(),
    #         list(range(params_np.shape[0])),
    #         param_idxs.reshape(-1).tolist(),
    #         scale_matrix.reshape(-1).tolist()
    #     )

    #     ref_grad, ref_hessians, ref_mps = derivatives.compute_ghm(nrg_op, x_ph, [params_tf])
    #     test_nrg, test_grads, test_hessians, test_mps = test_lj.total_derivative(x0, params_np.shape[0])

    #     sess = tf.Session()
    #     # np.testing.assert_array_almost_equal(test_nrg, sess.run(nrg_op, feed_dict={x_ph: x0}), decimal=13)
    #     np.testing.assert_allclose(test_grads, sess.run(ref_grad, feed_dict={x_ph: x0}), rtol=1e-10)
    #     # tighten the tolerance for this later.
    #     # np.testing.assert_array_almost_equal(test_hessians, sess.run(ref_hessians, feed_dict={x_ph: x0}), decimal=11)
    #     ref_h_val = sess.run(ref_hessians, feed_dict={x_ph: x0}).reshape(N*3, N*3)
    #     test_h_val = test_hessians.reshape(N*3, N*3)
    #     diff = np.tril(ref_h_val) - np.tril(test_h_val)
    #     # print("MAX DIFF", np.amax(diff), ref_h_val.reshape(-1)[np.argmax(diff)])
    #     np.testing.assert_allclose(np.tril(ref_h_val), np.tril(test_h_val), rtol=1e-10)
    #     np.testing.assert_allclose(test_mps, sess.run(ref_mps[0], feed_dict={x_ph: x0}), rtol=1e-9)


    def test_lj612_small_4d(self):

        x0 = np.array([
            [ 0.0637,   0.0126,   0.2203, 0.5],
            [ 1.0573,  -0.2011,   1.2864,-0.1],
            [ 2.3928,   1.2209,  -0.2230, 0.2],
            [-0.6891,   1.6983,   0.0780, 0.3],
            [-0.6312,  -1.6261,  -0.2601, 0.5]
        ], dtype=np.float64)

        params = np.array([3.0, 2.0, 1.0, 1.4], dtype=np.float64)
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

        box = np.array([
            [2.0, 0.5, 0.6],
            [0.6, 1.6, 0.3],
            [0.4, 0.7, 1.1]
        ], dtype=np.float64)

        energy_fn = functools.partial(nonbonded.lennard_jones,
            scale_matrix=scale_matrix,
            param_idxs=param_idxs,
            cutoff=None)

        check_grads(energy_fn, (x0, params, None), order=1, eps=1e-5)
        check_grads(energy_fn, (x0, params, None), order=2, eps=1e-7)


    def test_lj612_small(self):

        x0 = np.array([
            [ 0.0637,   0.0126,   0.2203],
            [ 1.0573,  -0.2011,   1.2864],
            [ 2.3928,   1.2209,  -0.2230],
            [-0.6891,   1.6983,   0.0780],
            [-0.6312,  -1.6261,  -0.2601]
        ], dtype=np.float64)

        params = np.array([3.0, 2.0, 1.0, 1.4], dtype=np.float64)
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

        box = np.array([
            [2.0, 0.5, 0.6],
            [0.6, 1.6, 0.3],
            [0.4, 0.7, 1.1]
        ], dtype=np.float64)

        energy_fn = functools.partial(nonbonded.lennard_jones,
            scale_matrix=scale_matrix,
            param_idxs=param_idxs,
            cutoff=None)

        assert_potential_invariance(energy_fn, x0, params, box)

        # check_grads(ref.energy, (x0, params), order=1)
        # check_grads(ref.energy, (x0, params), order=2)


        # check_grads(ref.energy, (x0, params, box), order=1, eps=1e-5)
        # check_grads(ref.energy, (x0, params, box), order=2, eps=1e-7)


if __name__ == "__main__":
    unittest.main()