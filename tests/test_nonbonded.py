import functools
import unittest
import scipy.linalg
from jax.config import config; config.update("jax_enable_x64", True)

import numpy as np
import jax
import jax.numpy as jnp

import functools

from common import GradientTest
from common import prepare_nonbonded_system, prepare_lj_system, prepare_es_system

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


    def test_electrostatics(self):
        np.random.seed(4321)
        D = 3

        test_system = self.get_random_coords(64, D)
        test_system = self.get_water_coords(D, sort=False)
        test_system = test_system[:2048]
        padding = 0.2
        diag = np.amax(test_system, axis=0) - np.amin(test_system, axis=0) + padding
        box = np.eye(3)
        np.fill_diagonal(box, diag)

        # _, test_system, box, _ = water_box.prep_system(8.1) # 6.2 is 23k atoms, roughly DHFR
        # _, test_system, box, _ = water_box.prep_system(6.2) # 6.2 is 23k atoms, roughly DHFR
        # test_system = test_system/test_system.unit

        # print("BOX:", box)

        # atom_idxs = np.arange(test_system.shape[0])
        # random_idxs = np.random.choice(atom_idxs, size=1024, replace=False)
        # test_system = test_system[random_idxs]

        # print("System shape", test_system.shape)

        # print("box", box)

        for coords in [test_system]:

            sort = True
            if sort:
                perm = hilbert_sort(coords+np.argmin(coords), D)
                coords = coords[perm]

            print(coords.shape)

            N = coords.shape[0]
            E = N//5

            lambda_offset_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)

            # for precision, rtol in [(np.float64, 1e-9), (np.float32, 5e-5)]:
            for precision, rtol in [(np.float64, 1e-9)]:
            # for precision, rtol in [(np.float32, 1e-2)]:

                for cutoff in [1.0]:
                    E = 0 # DEBUG!
                    charge_params, ref_potential, test_potential = prepare_es_system(
                        coords,
                        E,
                        lambda_offset_idxs,
                        p_scale=1.0,
                        cutoff=cutoff,
                        precision=precision
                    )

                    # for lamb in [0.0, 0.1, 0.2]:
                    for lamb in [0.05]*10:

                        print("lambda", lamb, "cutoff", cutoff, "precision", precision, "xshape", coords.shape)

                        self.compare_forces(
                            coords,
                            charge_params,
                            box,
                            lamb,
                            ref_potential,
                            test_potential,
                            precision,
                            rtol=rtol
                        )

    # @unittest.skip("temporary")
    # def test_lennard_jones(self):

    #     D = 3

    #     np.random.seed(1234)

    #     # test_system = self.get_random_coords(5, D)
    #     # test_system = self.get_random_coords(128, D)
    #     # test_system = self.get_random_coords(16, D)
    #     test_system = self.get_water_coords(D, sort=True)

    #     # padding = 0.3
    #     padding = 1000.0
    #     diag = np.amax(test_system, axis=0) - np.amin(test_system, axis=0) + padding
    #     box = np.eye(3)
    #     np.fill_diagonal(box, diag)

    #     for coords in [test_system]:

    #         N = coords.shape[0]
    #         E = N//5

    #         lambda_plane_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)
    #         lambda_offset_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)
    #         # lambda_group_idxs = np.random.randint(low=0, high=3, size=N, dtype=np.int32)

    #         # why are errors in du_dp so large?
    #         # error bars on single precision du/dp is pretty weird.
    #         # for precision, rtol in [(np.float64, 1e-9), (np.float32, 5e-5)]:
    #         # for precision, rtol in [(np.float64, 1e-9), (np.float32, 5e-5)]:
    #         for precision, rtol in [(np.float64, 1e-9)]:
    #         # for precision, rtol in [(np.float32, 1e-3)]:

    #             for cutoff in [100.0]:
    #                 # E = 0 # debug use ONLY
    #                 lj_params, ref_potential, test_potential = prepare_lj_system(
    #                     coords,
    #                     E,
    #                     lambda_plane_idxs,
    #                     lambda_offset_idxs,
    #                     p_scale=10.0,
    #                     tip3p=True,
    #                     cutoff=cutoff,
    #                     precision=precision
    #                 )

    #                 # ref_nb_parameterized = functools.partial(ref_nb,
    #                 #     lj_params=lj_params
    #                 # )
    #                 # non periodic for now - switch to periodic next!

    #                 # for lamb in [0.0, 0.1, 0.2]:
    #                 for lamb in [0.0]:

    #                     print("lambda", lamb, "cutoff", cutoff, "precision", precision)

    #                     self.compare_forces(
    #                         coords,
    #                         lj_params,
    #                         box,
    #                         lamb,
    #                         ref_potential,
    #                         test_potential,
    #                         precision,
    #                         rtol=rtol
    #                     )


    # def test_fast_nonbonded(self):
    #     np.random.seed(4321)
    #     D = 3


    #     # small_system = self.get_random_coords(43, D)
    #     large_system = self.get_water_coords(D, sort=True)

    #     for x_primal in [large_system]:

    #         N = x_primal.shape[0]
    #         E = N//5

    #         lambda_plane_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)
    #         lambda_offset_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)

    #         x_tangent = np.random.randn(*x_primal.shape)
    #         lamb_tangent = np.random.rand()

    #         for precision, rtol in [(np.float64, 1e-9), (np.float32, 5e-5)]:

    #             for cutoff in [10000.0]:
    #                 # E = 0 # DEBUG!
    #                 (charge_params, lj_params), ref_nb, test_force_ctor = prepare_nonbonded_system(
    #                     x_primal,
    #                     E,
    #                     lambda_plane_idxs,
    #                     lambda_offset_idxs,
    #                     p_scale=10.0,
    #                     cutoff=cutoff,
    #                     precision=precision
    #                 )

    #                 ref_nb_parameterized = functools.partial(ref_nb,
    #                     charge_params=charge_params,
    #                     lj_params=lj_params
    #                 )

    #                 for lamb_primal in [0.0, 0.1, 0.2]:

    #                     test_force = test_force_ctor()

    #                     print("lambda", lamb_primal, "cutoff", cutoff, "precision", precision, "xshape", x_primal.shape)

    #                     self.compare_forces(
    #                         x_primal,
    #                         lamb_primal,
    #                         x_tangent,
    #                         lamb_tangent,
    #                         ref_nb_parameterized,
    #                         test_force,
    #                         precision,
    #                         rtol=rtol
    #                     )

    #                     primals = (x_primal, lamb_primal, charge_params, lj_params)
    #                     tangents = (x_tangent, lamb_tangent, np.zeros_like(charge_params), np.zeros_like(lj_params))

    #                     grad_fn = jax.grad(ref_nb, argnums=(0, 1, 2, 3))
    #                     ref_primals, ref_tangents = jax.jvp(grad_fn, primals, tangents)

    #                     ref_du_dcharge_primals = ref_primals[2]
    #                     test_du_dcharge_primals = test_force.get_du_dcharge_primals()
    #                     np.testing.assert_almost_equal(ref_du_dcharge_primals, test_du_dcharge_primals, rtol)

    #                     ref_du_dcharge_tangents = ref_tangents[2]
    #                     test_du_dcharge_tangents = test_force.get_du_dcharge_tangents()
    #                     np.testing.assert_almost_equal(ref_du_dcharge_tangents, test_du_dcharge_tangents, rtol)

    #                     ref_du_dlj_primals = ref_primals[3]
    #                     test_du_dlj_primals = test_force.get_du_dlj_primals()
    #                     np.testing.assert_almost_equal(ref_du_dlj_primals, test_du_dlj_primals, rtol)

    #                     ref_du_dlj_tangents = ref_tangents[3]
    #                     test_du_dlj_tangents = test_force.get_du_dlj_tangents()
    #                     np.testing.assert_almost_equal(ref_du_dlj_tangents, test_du_dlj_tangents, rtol)


if __name__ == "__main__":
    unittest.main()
