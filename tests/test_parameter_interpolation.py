from jax.config import config; config.update("jax_enable_x64", True)

import functools
import numpy as np

from common import GradientTest
from timemachine.lib import potentials

from common import prepare_water_system

def interpolated_potential(conf, params, box, lamb, u_fn):
    assert params.size % 2 == 0

    CP = params.shape[0]//2
    new_params = (1-lamb)*params[:CP] + lamb*params[CP:] 

    return u_fn(conf, new_params, box, lamb)


class TestInterpolatedPotential(GradientTest):

    def test_nonbonded(self):

        # we test nonbonded terms to ensure that we're doing the chain rule with du_dl correctly since
        # bonded terms do not depend on lambda.

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

        for lamb in [0.0, 0.2, 1.0]:
            for precision, rtol in [(np.float64, 1e-8), (np.float32, 1e-4)]:

                    # E = 0 # DEBUG!
                    qlj_src, ref_potential, test_potential = prepare_water_system(
                        coords,
                        lambda_plane_idxs,
                        lambda_offset_idxs,
                        p_scale=1.0,
                        cutoff=cutoff
                    )

                    qlj_dst, _, _ = prepare_water_system(
                        coords,
                        lambda_plane_idxs,
                        lambda_offset_idxs,
                        p_scale=1.0,
                        cutoff=cutoff
                    )

                    qlj = np.concatenate([qlj_src, qlj_dst])

                    print("lambda", lamb, "cutoff", cutoff, "precision", precision, "xshape", coords.shape)

                    ref_potential = functools.partial(
                        interpolated_potential,
                        u_fn=ref_potential)

                    test_potential = potentials.InterpolatedPotential(
                        test_potential,
                        N, qlj.size,
                    )

                    self.compare_forces(
                        coords,
                        qlj,
                        box,
                        lamb,
                        ref_potential,
                        test_potential,
                        rtol,
                        precision=precision
                    )