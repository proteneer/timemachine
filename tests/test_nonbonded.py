import functools
import unittest
import scipy.linalg
from jax.config import config; config.update("jax_enable_x64", True)

import numpy as np
import jax
import jax.numpy as jnp

import functools

from common import GradientTest
from common import prepare_nonbonded_system, prepare_lj_system, prepare_nb_system

from timemachine.lib import potentials

from training import water_box

from hilbertcurve.hilbertcurve import HilbertCurve

np.set_printoptions(linewidth=500)


def hilbert_sort(conf, D):
    hc = HilbertCurve(64, D)
    int_confs = (conf*1000).astype(np.int64)
    dists = []
    for xyz in int_confs.tolist():
        dist = hc.distance_from_coordinates(xyz)
        dists.append(dist)
    perm = np.argsort(dists)
    return perm


class TestNonbonded(GradientTest):


    def test_nonbonded(self):

        np.random.seed(4321)
        D = 3

        benchmark = False

        # for size in [32, 230, 1051]:
        for size in [32, 230, 1051]:

            if not benchmark:
                water_coords = self.get_water_coords(D, sort=False)
                test_system = water_coords[:size]
                test_system[1] = test_system[0]+1e-5 # very delta epsilon to trigger a singularity
                padding = 0.2
                diag = np.amax(test_system, axis=0) - np.amin(test_system, axis=0) + padding
                box = np.eye(3)
                np.fill_diagonal(box, diag)
            else:
                # _, test_system, box, _ = water_box.prep_system(8.1) # 8.1 is 50k atoms, roughly DHFR
                _, test_system, box, _ = water_box.prep_system(6.2) # 6.2 is 23k atoms, roughly DHFR
                test_system = test_system/test_system.unit

            for coords in [test_system]:

                N = coords.shape[0]
                # E = N//5

                lambda_offset_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)

                for precision, rtol in [(np.float64, 1e-9), (np.float32, 1e-4)]:
                # for precision, rtol in [(np.float32, 1e-4)]:

                    for cutoff in [1.0]:
                        E = 1 # DEBUG!
                        charge_params, ref_potential, test_potential = prepare_nb_system(
                            coords,
                            E,
                            lambda_offset_idxs,
                            p_scale=1.0,
                            cutoff=cutoff,
                            precision=precision
                        )

                        for lamb in [0.0, 0.1, 0.2]:
                        # for lamb in [0.0]:

                            print("lambda", lamb, "cutoff", cutoff, "precision", precision, "xshape", coords.shape)

                            self.compare_forces(
                                coords,
                                charge_params,
                                box,
                                lamb,
                                ref_potential,
                                test_potential,
                                precision,
                                rtol=rtol,
                                benchmark=benchmark
                            )


if __name__ == "__main__":
    unittest.main()
