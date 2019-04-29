import unittest
import numpy as np
from timemachine.jax_functionals import jax_utils

def reference_periodic_distance(ri, rj, box):
    diff = rj - ri
    diff -= box[2]*np.floor(diff[2]/box[2][2]+0.5);
    diff -= box[1]*np.floor(diff[1]/box[1][1]+0.5);
    diff -= box[0]*np.floor(diff[0]/box[0][0]+0.5);
    return np.linalg.norm(diff)

class TestJaxUtils(unittest.TestCase):

    def test_bonded_periodic_distance(self):
        conf = np.array([
            [ 0.0637,   0.0126,   0.2203], # C
            [ 1.0573,  -0.2011,   1.2864], # H
            [ 2.3928,   1.2209,  -0.2230], # H
            [-0.6891,   1.6983,   0.0780], # H
            [-0.6312,  -1.6261,  -0.2601], # H
        ], dtype=np.float64)

        box = np.array([
            [1.3, 0.5, 0.6],
            [0.6, 1.2, 0.45],
            [0.4, 0.3, 1.2]
        ], dtype=np.float64)

        src_idxs = [0,1,3,2]
        dst_idxs = [1,2,0,1]

        ri = conf[src_idxs]
        rj = conf[dst_idxs]

        dsts = jax_utils.distance(ri, rj, box)
        for idx, (i,j) in enumerate(zip(src_idxs, dst_idxs)):
            dij = reference_periodic_distance(conf[i], conf[j], box)
            np.testing.assert_array_almost_equal(dij, dsts[idx])

    def test_pairwise_periodic_distance(self):
        conf = np.array([
            [-3.7431,    0.0007,    3.3896],
            [-2.2513,    0.2400,    3.0656],
            [-1.8353,   -0.3114,    1.6756],
            [-0.3457,   -0.0141,    1.3424],
            [ 0.0913,   -0.5912,   -0.0304],
            [ 1.5230,   -0.3150,   -0.3496],
            [ 2.4800,   -1.3537,   -0.3510],
            [ 3.8242,   -1.1091,   -0.6908],
            [ 4.2541,    0.2018,   -1.0168],
            [ 3.2918,    1.2461,   -1.0393],
            [ 1.9507,    0.9887,   -0.6878],
            [ 3.6069,    2.5311,   -1.4305],
            [ 4.6870,    2.6952,   -2.3911],
            [ 5.9460,    1.9841,   -1.7744],
            [ 5.6603,    0.4771,   -1.4483],
            [ 6.7153,    0.0454,   -0.5274],
            [ 8.0153,    0.3238,   -0.7754],
            [ 8.3940,    1.0806,   -1.9842],
            [ 7.3027,    2.0609,   -2.5505],
            [ 9.0311,   -0.1319,    0.1662],
            [ 4.2434,    2.1598,   -3.7921],
            [ 4.9088,    4.2364,   -2.4878],
            [ 4.6917,   -2.1552,   -0.7266],
            [-3.9733,    0.4081,    4.3758],
            [-3.9690,   -1.0674,    3.3947],
            [-4.3790,    0.4951,    2.6522],
            [-1.6465,   -0.2405,    3.8389],
            [-2.0559,    1.3147,    3.1027],
            [-1.9990,   -1.3929,    1.6608],
            [-2.4698,    0.1371,    0.9054],
            [-0.1921,    1.0695,    1.3452],
            [ 0.2880,   -0.4452,    2.1239],
            [-0.0916,   -1.6686,   -0.0213],
            [-0.5348,   -0.1699,   -0.8201],
            [ 2.2004,   -2.3077,   -0.1063],
            [ 1.2776,    1.7607,   -0.6991],
            [ 6.1198,    2.5014,   -0.8189],
            [ 5.7881,   -0.1059,   -2.3685],
            [ 6.4691,   -0.4538,    0.2987],
            [ 9.3048,    1.6561,   -1.8023],
            [ 8.6369,    0.3417,   -2.7516],
            [ 7.6808,    3.0848,   -2.5117],
            [ 7.1355,    1.8275,   -3.6048],
            [ 8.8403,    0.2961,    1.1526],
            [10.0386,    0.1617,   -0.1353],
            [ 9.0076,   -1.2205,    0.2406],
            [ 4.1653,    1.0737,   -3.8000],
            [ 4.9494,    2.4548,   -4.5696],
            [ 3.2631,    2.5647,   -4.0515],
            [ 3.9915,    4.7339,   -2.8073],
            [ 5.1949,    4.6493,   -1.5175],
            [ 5.6935,    4.4750,   -3.2076],
            [ 4.1622,   -2.9559,   -0.5467]
        ], dtype=np.float64)

        N = conf.shape[0]

        box = np.array([
            [1.3, 0.5, 0.6],
            [0.6, 1.2, 0.45],
            [0.4, 0.3, 1.2]
        ], dtype=np.float64)

        ri = np.expand_dims(conf, 0)
        rj = np.expand_dims(conf, 1)

        dij = jax_utils.distance(ri, rj, box)

        for i in range(N):
            for j in range(N):
                expected = reference_periodic_distance(conf[i], conf[j], box)
                np.testing.assert_array_almost_equal(dij[i][j], expected)

if __name__ == "__main__":
    unittest.main()