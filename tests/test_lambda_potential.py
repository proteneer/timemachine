from jax.config import config; config.update("jax_enable_x64", True)

import functools
import numpy as np

from common import GradientTest
from timemachine.lib import potentials

from common import prepare_water_system


def lambda_potential(conf, params, box, lamb, sign, u_fn):
    """
    This would be more useful if we could also do (1-lamb)
    """

    # lamb appears twice as the potential itself may be a function
    # of lambda (eg. if we use softcores)
    return sign * lamb * u_fn(conf, params, box, lamb)


class TestLambdaPotential(GradientTest):

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

        for sign in [2, 1, -1, 3]:

            for precision, rtol in [(np.float64, 1e-8), (np.float32, 1e-4)]:

                    # E = 0 # DEBUG!
                    charge_params, ref_potential, test_potential = prepare_water_system(
                        coords,
                        lambda_plane_idxs,
                        lambda_offset_idxs,
                        p_scale=1.0,
                        cutoff=cutoff,
                        precision=precision
                    )

                    lamb = 0.2

                    print("lambda", lamb, "cutoff", cutoff, "precision", precision, "xshape", coords.shape)

                    ref_potential = functools.partial(lambda_potential, sign=sign, u_fn=ref_potential)
                    test_potential = potentials.LambdaPotential(
                        test_potential,
                        N,charge_params.size,
                        sign
                    )

                    self.compare_forces(
                        coords,
                        charge_params,
                        box,
                        lamb,
                        ref_potential,
                        test_potential,
                        rtol
                    )