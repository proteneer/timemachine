from jax.config import config

config.update("jax_enable_x64", True)

import numpy as np
import pytest
from common import GradientTest, prepare_water_system

from timemachine.potentials import nonbonded

pytestmark = [pytest.mark.memcheck]


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
        box = np.eye(3) * 3

        N = coords.shape[0]

        lambda_plane_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)
        lambda_offset_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)

        sigma_scale = 5.0

        qlj_src, potential = prepare_water_system(
            coords, lambda_plane_idxs, lambda_offset_idxs, p_scale=sigma_scale, cutoff=cutoff
        )

        qlj_dst, _ = prepare_water_system(
            coords, lambda_plane_idxs, lambda_offset_idxs, p_scale=sigma_scale, cutoff=cutoff
        )

        qlj = np.concatenate([qlj_src, qlj_dst])

        ref_interpolated_potential = nonbonded.interpolated(potential.to_reference())

        test_interpolated_potential = potential.to_gpu().interpolate()

        lambda_vals = [0.0, 0.2, 1.0]

        for precision, rtol, atol in [(np.float64, 1e-8, 3e-11), (np.float32, 1e-4, 3e-6)]:
            # E = 0 # DEBUG!
            print("cutoff", cutoff, "precision", precision, "xshape", coords.shape)
            self.compare_forces(
                coords,
                qlj,
                box,
                lambda_vals,
                ref_interpolated_potential,
                test_interpolated_potential,
                rtol=rtol,
                atol=atol,
                precision=precision,
            )
