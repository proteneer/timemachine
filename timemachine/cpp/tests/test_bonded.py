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


    def test_restraint(self):

        B = 8

        params = np.random.randn(B, 3)

        N = 10
        D = 3

        b_idxs = []

        for _ in range(B):
            b_idxs.append(np.random.choice(np.arange(N), size=2, replace=False))

        b_idxs = np.array(b_idxs, dtype=np.int32)

        lambda_flags = np.random.randint(0, 2, size=(B,))

        for precision, rtol in [(np.float32, 2e-5), (np.float64, 1e-9)]:

            ref_nrg = jax.partial(
                bonded.restraint,
                lamb_flags=lambda_flags,
                box=None,
                bond_idxs=b_idxs
            )



            ref_nrg_params = jax.partial(
                ref_nrg,
                params=params
            )

            x_primal = self.get_random_coords(N, D)
            x_tangent = np.random.randn(*x_primal.shape)
            lamb_tangent = np.random.rand()

            for lamb_primal in [0.0, 0.1, 0.5, 0.7, 1.0]:

                # we need to clear the du_dp buffer each time, so we need
                # to instantiate test_nrg inside here
                test_nrg = ops.Restraint(
                    np.array(b_idxs, dtype=np.int32),
                    np.array(params, dtype=np.float64),
                    np.array(lambda_flags, dtype=np.int32),
                    precision=precision
                )

                self.compare_forces(
                    x_primal,
                    lamb_primal,
                    x_tangent,
                    lamb_tangent,
                    ref_nrg_params,
                    test_nrg,
                    precision,
                    rtol
                )

                primals = (x_primal, lamb_primal, params)
                tangents = (x_tangent, lamb_tangent, np.zeros_like(params))

                grad_fn = jax.grad(ref_nrg, argnums=(0, 1, 2))
                ref_primals, ref_tangents = jax.jvp(grad_fn, primals, tangents)

                ref_du_dp_primals = ref_primals[2]
                test_du_dp_primals = test_nrg.get_du_dp_primals()
                np.testing.assert_almost_equal(ref_du_dp_primals, test_du_dp_primals, rtol)

                ref_du_dp_tangents = ref_tangents[2]
                test_du_dp_tangents = test_nrg.get_du_dp_tangents()
                np.testing.assert_almost_equal(ref_du_dp_tangents, test_du_dp_tangents, rtol)


    def test_bonded(self):
        np.random.seed(125)

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

        atom_idxs = np.arange(N)
        bond_params = np.random.rand(B, 2).astype(np.float64)
        bond_idxs = []
        for _ in range(B):
            bond_idxs.append(np.random.choice(atom_idxs, size=2, replace=False))
        bond_idxs = np.array(bond_idxs, dtype=np.int32)

        lamb = 0.0

        for precision, rtol in [(np.float32, 2e-5), (np.float64, 1e-9)]:

            custom_bonded = ops.HarmonicBond(
                bond_idxs,
                bond_params,
                precision=precision
            )

            # test the parameter derivatives for correctness.
            harmonic_bond_fn = functools.partial(bonded.harmonic_bond, box=None, bond_idxs=bond_idxs)
            grad_fn = jax.grad(harmonic_bond_fn, argnums=(0, 1, 2))

            harmonic_bond_fn_params = functools.partial(
                harmonic_bond_fn,
                params=bond_params
            )

            x_tangent = np.random.randn(*x.shape)
            lamb_tangent = np.random.rand()

            self.compare_forces(
                x,
                lamb,
                x_tangent,
                lamb_tangent,
                harmonic_bond_fn_params,
                custom_bonded,
                precision,
                rtol
            )

            primals = (x, lamb, bond_params)
            tangents = (x_tangent, lamb_tangent, np.zeros_like(bond_params))

            ref_primals, ref_tangents = jax.jvp(grad_fn, primals, tangents)

            ref_du_dp_primals = ref_primals[2]
            test_du_dp_primals = custom_bonded.get_du_dp_primals()
            np.testing.assert_almost_equal(ref_du_dp_primals, test_du_dp_primals, rtol)

            ref_du_dp_tangents = ref_tangents[2]
            test_du_dp_tangents = custom_bonded.get_du_dp_tangents()
            np.testing.assert_almost_equal(ref_du_dp_tangents, test_du_dp_tangents, rtol)

            # compre du_dps

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
