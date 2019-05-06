import functools
import unittest
import numpy as np

from jax.config import config; config.update("jax_enable_x64", True)
from jax.test_util import check_grads

from timemachine.jax_functionals import jax_gbsa

class TestGBSA(unittest.TestCase):

    def test_gbsa(self):
        """ Testing the GBSA OBC model. """
        conf = np.array([
            [ 0.0637,   0.0126,   0.2203],
            [ 1.0573,  -0.2011,   1.2864],
            [ 2.3928,   1.2209,  -0.2230],
            [-0.6891,   1.6983,   0.0780],
            [-0.6312,  -1.6261,  -0.2601]
        ], dtype=np.float64)

        params = np.array([
            .1984, .115, .85, # H
            0.0221, .19, .72  # C
        ])

        param_idxs = np.array([
            [3, 4, 5],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ])

        gb_nrg = jax_gbsa.GBSA(param_idxs)

        atomic_radii = params[param_idxs[:, 1]]
        scale_factors = params[param_idxs[:, 2]]

        gb_radii = gb_nrg.born_radii(conf, atomic_radii, scale_factors)

        check_grads(gb_nrg.born_radii, (conf, atomic_radii, scale_factors), order=1)
        check_grads(gb_nrg.born_radii, (conf, atomic_radii, scale_factors), order=2)

        gb_e = gb_nrg.energy(conf, params)

        check_grads(gb_nrg.energy, (conf, params), order=1, eps=1e-6)
        check_grads(gb_nrg.energy, (conf, params), order=2, eps=1e-8)

if __name__ == "__main__":
    unittest.main()