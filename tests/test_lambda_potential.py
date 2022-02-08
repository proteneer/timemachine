import unittest

from jax.config import config

config.update("jax_enable_x64", True)

import functools

import numpy as np
from common import GradientTest, prepare_water_system

from timemachine.lib import potentials


def lambda_potential(conf, params, box, lamb, multiplier, offset, u_fn):
    """
    Implements:

    (multiplier*lamb + offset)*u_fn(lamb)

    For example, to implement (1-lambda)*U_0(lambda) + lambda*U_1(lambda),
    The left hand side is (multiplier = 1, offset = 0),
    The right hand side is (multiplier = -1, offset = 1)

    """

    # lamb appears twice as the potential itself may be a function
    # of lambda (eg. if we use softcores)
    return (multiplier * lamb + offset) * u_fn(conf, params, box, lamb)


class TestLambdaPotential(GradientTest):
    @unittest.skip("not supported")
    def test_nonbonded(self):

        np.random.seed(4321)
        D = 3

        cutoff = 1.0
        size = 36

        water_coords = self.get_water_coords(D, sort=False)
        coords = water_coords[:size]
        padding = 0.2
        diag = np.amax(coords, axis=0) - np.amin(coords, axis=0) + padding
        box = np.eye(3)
        np.fill_diagonal(box, diag)

        N = coords.shape[0]

        lambda_plane_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)
        lambda_offset_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)

        for multiplier in [-2.2, 0.0, 0.3]:
            for offset in [0.0, 0.2, -2.3]:
                for lamb in [0.0, 0.2, 1.5]:
                    for precision, rtol in [(np.float64, 1e-8), (np.float32, 1e-4)]:

                        # E = 0 # DEBUG!
                        params, ref_potential, test_potential = prepare_water_system(
                            coords, lambda_plane_idxs, lambda_offset_idxs, p_scale=1.0, cutoff=cutoff
                        )

                        print(
                            "multiplier",
                            multiplier,
                            "lambda",
                            lamb,
                            "cutoff",
                            cutoff,
                            "precision",
                            precision,
                            "xshape",
                            coords.shape,
                            "offset",
                            offset,
                        )

                        ref_potential = functools.partial(
                            lambda_potential, multiplier=multiplier, offset=offset, u_fn=ref_potential
                        )

                        test_potential = potentials.LambdaPotential(
                            test_potential,
                            N,
                            params.size,
                            multiplier,
                            offset,
                        )

                        self.compare_forces(
                            coords, params, box, lamb, ref_potential, test_potential, rtol, precision=precision
                        )
