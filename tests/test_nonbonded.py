import functools
import unittest
import scipy.linalg
from jax.config import config; config.update("jax_enable_x64", True)

import numpy as np
import jax
import jax.numpy as jnp

import functools

from common import GradientTest
from common import prepare_nb_system, prepare_water_system

from timemachine.potentials import bonded, nonbonded, gbsa
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

    @unittest.skip("boo")
    def test_exclusion(self):

        # This test verifies behavior when two particles are arbitrarily
        # close but are marked as excluded to ensure proper cancellation
        # of exclusions occur in the fixed point math.

        np.random.seed(2020)

        # water_coords = self.get_water_coords(3, sort=False)


        water_coords = self.get_water_coords(3, sort=False)
        test_system = water_coords[:126] # multiple of 3
        padding = 0.2
        diag = np.amax(test_system, axis=0) - np.amin(test_system, axis=0) + padding
        box = np.eye(3)
        np.fill_diagonal(box, diag)

        N = test_system.shape[0]

        EA = 0

        atom_idxs = np.arange(test_system.shape[0])

        # pick a set of atoms that will be mutually excluded from each other.
        # we will need to set their exclusions manually
        # exclusion_atoms = np.random.choice(atom_idxs, size=(EA), replace=False)
        # exclusion_idxs = []

        # for idx, i in enumerate(exclusion_atoms):
        #     for jdx, j in enumerate(exclusion_atoms):
        #         if jdx > idx:
        #             exclusion_idxs.append((i,j))

        # E = len(exclusion_idxs)

        # print(exclusion_idxs)

        # exclusion_idxs = np.array(exclusion_idxs, dtype=np.int32)
        # scales = np.ones((E, 2), dtype=np.float64)
        # perturb the system
        # for idx in exclusion_atoms:
            # test_system[idx] = np.zeros(3) + np.random.rand()/1000+2

        # water exclusions

        # exclusion_idxs = []
        # for i in range(N//3):
        #     O_idx = i*3+0
        #     H1_idx = i*3+1
        #     H2_idx = i*3+2
        #     exclusion_idxs.append([O_idx, H1_idx]) # 1-2
        #     exclusion_idxs.append([O_idx, H2_idx]) # 1-2
        #     exclusion_idxs.append([H1_idx, H2_idx]) # 1-3
        # exclusion_idxs = np.array(exclusion_idxs, dtype=np.int32)
        # E = len(exclusion_idxs)
        # scales = np.ones((E, 2), dtype=np.float64)

        beta = 2.0

        lambda_offset_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)

        cutoff = 1.0

        # for precision, rtol in [(np.float64, 1e-9), (np.float32, 1e-4)]:
        for precision, rtol in [(np.float64, 1e-9)]:

            test_u = potentials.Nonbonded(
                exclusion_idxs,
                scales,
                lambda_offset_idxs,
                beta,
                cutoff,
                precision=precision
            )

            charge_rescale_mask = np.ones((N, N))
            for (i,j), exc in zip(exclusion_idxs, scales[:, 0]):
                charge_rescale_mask[i][j] = 1 - exc
                charge_rescale_mask[j][i] = 1 - exc

            lj_rescale_mask = np.ones((N, N))
            for (i,j), exc in zip(exclusion_idxs, scales[:, 1]):
                lj_rescale_mask[i][j] = 1 - exc
                lj_rescale_mask[j][i] = 1 - exc

            ref_u = functools.partial(
                nonbonded.nonbonded_v3,
                charge_rescale_mask=charge_rescale_mask,
                lj_rescale_mask=lj_rescale_mask,
                scales=scales,
                beta=beta,
                cutoff=cutoff,
                lambda_offset_idxs=lambda_offset_idxs
            )

            lamb = 0.0

            params = np.stack([
                (np.random.rand(N).astype(np.float64) - 0.5)*np.sqrt(138.935456), # q
                np.random.rand(N).astype(np.float64)/3.0, # sig
                np.random.rand(N).astype(np.float64) # eps
            ], axis=1)

            self.compare_forces(
                test_system,
                params,
                box,
                lamb,
                ref_u,
                test_u,
                precision,
                rtol=rtol,
                benchmark=False
            )

    def test_nonbonded(self):

        np.random.seed(4321)
        D = 3

        benchmark = False

        # test numerical accuracy on a box of water

        for size in [33, 231, 1050]:

            if not benchmark:
                water_coords = self.get_water_coords(D, sort=False)
                test_system = water_coords[:size]
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

                lambda_offset_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)

                for precision, rtol in [(np.float64, 1e-9), (np.float32, 3e-4)]:
                # for precision, rtol in [(np.float32, 1e-4)]:

                    for cutoff in [1.0]:
                        # E = 0 # DEBUG!
                        charge_params, ref_potential, test_potential = prepare_water_system(
                            coords,
                            lambda_offset_idxs,
                            p_scale=1.0,
                            cutoff=cutoff,
                            precision=precision
                        )

                        for lamb in [0.0, 0.1, 0.2]:

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
