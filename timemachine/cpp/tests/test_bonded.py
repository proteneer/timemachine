import functools
import unittest
import scipy.linalg

import numpy as np
import jax
import jax.numpy as jnp
from jax.config import config; config.update("jax_enable_x64", True)
import functools

from common import GradientTest
from common import prepare_bonded_system
from timemachine.lib import ops
from timemachine.potentials import alchemy, bonded

class TestBonded(GradientTest):


    def test_flat_bottom(self):

        B = 10

        P = 24
        params = np.random.rand(24)
        p_idxs = np.random.randint(0, P, size=(B, 2))

        N = 50
        D = 3
        x = self.get_random_coords(N, D)

        b_idxs = []

        for _ in range(B):
            b_idxs.append(np.random.choice(np.arange(N), size=2, replace=False))

        b_idxs = np.array(b_idxs, dtype=np.int32)

        lambda_flags = np.random.randint(0, 2, size=(B,))


        for precision, rtol in [(np.float32, 2e-5), (np.float64, 1e-9)]:


            ref_nrg = jax.partial(
                bonded.flat_bottom,
                lamb_flags=lambda_flags,
                box=None,
                bond_idxs=b_idxs,
                param_idxs = p_idxs
            )

            test_nrg = ops.FlatBottom(
                np.array(b_idxs, dtype=np.int32),
                np.array(p_idxs, dtype=np.int32),
                np.array(lambda_flags, dtype=np.int32),
                0,
                precision=precision
            )

            for lamb in [0.1, 0.0, 0.1, 0.5, 0.7, 1.0]:

                print("precision", precision, "lambda", lamb)

                self.compare_forces(
                    x,
                    params,
                    lamb,
                    ref_nrg,
                    test_nrg,
                    precision=precision,
                    rtol=rtol
                )


    # def test_bonded(self):
    #     np.random.seed(125)

    #     P_bonds = 4
    #     P_angles = 6
    #     P_torsions = 13
 
    #     N = 64
    #     B = 35
    #     A = 36
    #     T = 37

    #     # N = 4
    #     # B = 6
    #     # A = 1
    #     # T = 1

    #     D = 3

    #     x = self.get_random_coords(N, D)

    #     for precision, rtol in [(np.float32, 2e-5), (np.float64, 1e-9)]:

    #         params, ref_bonds, custom_bonds = prepare_bonded_system(
    #             x,
    #             P_bonds,
    #             P_angles,
    #             P_torsions,
    #             B,
    #             A,
    #             T,
    #             precision
    #         )

    #         for lamb in [0.0, 0.4, 0.5, 1.0]:
    #             for r, t in zip(ref_bonds, custom_bonds):
    #                 self.compare_forces(
    #                     x,
    #                     params,
    #                     lamb,
    #                     r,
    #                     t,
    #                     precision,
    #                     rtol
    #                 )

    # def test_alchemical_bonded(self):
    #     np.random.seed(125)

    #     P_bonds = 4
    #     P_angles = 6
    #     P_torsions = 13
 
    #     N = 64
    #     B = 35
    #     A = 36
    #     T = 37

    #     # N = 4
    #     # B = 6
    #     # A = 1
    #     # T = 1

    #     D = 3

    #     x = self.get_random_coords(N, D)

    #     # for precision, rtol in [(np.float32, 2e-5), (np.float64, 1e-9)]:

    #     for precision, rtol in [(np.float64, 1e-9), (np.float32, 2e-5)]:


    #         params, ref_bonds0, custom_bonds0 = prepare_bonded_system(
    #             x,
    #             P_bonds,
    #             P_angles,
    #             P_torsions,
    #             B,
    #             A,
    #             T,
    #             precision
    #         )

    #         params, ref_bonds1, custom_bonds1 = prepare_bonded_system(
    #             x,
    #             P_bonds,
    #             P_angles,
    #             P_torsions,
    #             B,
    #             A,
    #             T,
    #             precision,
    #             params
    #         )

    #         terms = len(ref_bonds0)

    #         for idx in range(terms):
    #             ref_fn = functools.partial(
    #                 alchemy.linear_rescale,
    #                 fn0 = ref_bonds0[idx],
    #                 fn1 = ref_bonds1[idx]
    #             )

    #             test_fn = ops.AlchemicalGradient(
    #                 N,
    #                 len(params),
    #                 custom_bonds0[idx],
    #                 custom_bonds1[idx]
    #             )

    #             for lamb in [0.0, 0.4, 0.5, 1.0]:
    #                 # print("LAMBDA", lamb, "EXPONENT", exponent)
    #                 # for r, t in zip(ref_bonds, custom_bonds):
    #                 self.compare_forces(
    #                     x,
    #                     params,
    #                     lamb,
    #                     ref_fn,
    #                     test_fn,
    #                     precision,
    #                     rtol
    #                 )
