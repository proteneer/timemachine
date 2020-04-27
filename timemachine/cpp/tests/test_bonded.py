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

class TestBonded(GradientTest):

    def test_bonded(self):
        np.random.seed(125)

        P_bonds = 4
        P_angles = 6
        P_torsions = 13
 
        N = 64
        B = 35
        A = 36
        T = 37

        # N = 4
        # B = 6
        # A = 1
        # T = 1

        D = 3

        x = self.get_random_coords(N, D)

        for precision, rtol in [(np.float32, 2e-5), (np.float64, 1e-9)]:

            params, ref_bonds, custom_bonds = prepare_bonded_system(
                x,
                P_bonds,
                P_angles,
                P_torsions,
                B,
                A,
                T,
                precision
            )

            for lamb in [0.0, 0.4, 0.5, 1.0]:
                print("LAMBDA", lamb)
                for r, t in zip(ref_bonds, custom_bonds):
                    self.compare_forces(
                        x,
                        params,
                        lamb,
                        r,
                        t,
                        precision,
                        rtol
                    )
