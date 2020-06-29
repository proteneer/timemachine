import functools
import unittest
import scipy.linalg
from jax.config import config; config.update("jax_enable_x64", True)

import numpy as np
import jax
import jax.numpy as jnp

import functools

from common import GradientTest
from common import prepare_nonbonded_system

from timemachine.lib import custom_ops
from timemachine.potentials import alchemy
from timemachine.lib import ops, custom_ops

np.set_printoptions(linewidth=500)

class TestNonbonded(GradientTest):

    def get_water_system(self,
        D,
        P_charges,
        P_lj,
        sort=False):

        x = np.load("water.npy").astype(np.float64)
        if sort:
            perm = hilbert_sort(x, D)
            x = x[perm, :]

        N = x.shape[0]

        params = np.random.rand(P_charges).astype(np.float64)
        params = np.zeros_like(params)
        param_idxs = np.random.randint(low=0, high=P_charges, size=(N), dtype=np.int32)

        lj_params = np.random.rand(P_lj)/10 # we want these to be pretty small for numerical stability reasons
        lj_param_idxs = np.random.randint(low=0, high=P_lj, size=(N,2), dtype=np.int32)
        lj_param_idxs = lj_param_idxs + len(params) # offset 

        return x, np.concatenate([params, lj_params]), param_idxs, lj_param_idxs

    def get_ref_mp(self, x, params, param_idxs, cutoff):
        ref_mp = mixed_fn(x, params, param_idxs, cutoff)[0][0]
        ref_mp = np.transpose(ref_mp, (2,0,1))
        return ref_mp

    def test_fast_nonbonded(self):
        np.random.seed(4321)
        D = 3


        # small_system = self.get_random_coords(43, D)
        large_system = self.get_water_coords(D, sort=True)

        for x_primal in [large_system]:

            N = x_primal.shape[0]
            E = N//5

            lambda_plane_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)
            lambda_offset_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)

            x_tangent = np.random.randn(*x_primal.shape)
            lamb_tangent = np.random.rand()

            for precision, rtol in [(np.float64, 1e-9), (np.float32, 5e-5)]:

                for cutoff in [10000.0]:
                    # E = 0 # DEBUG!
                    (charge_params, lj_params), ref_nb, test_force_ctor = prepare_nonbonded_system(
                        x_primal,
                        E,
                        lambda_plane_idxs,
                        lambda_offset_idxs,
                        p_scale=10.0,
                        cutoff=cutoff,
                        precision=precision
                    )

                    ref_nb_parameterized = functools.partial(ref_nb,
                        charge_params=charge_params,
                        lj_params=lj_params
                    )

                    for lamb_primal in [0.0, 0.1, 0.2]:

                        test_force = test_force_ctor()

                        print("lambda", lamb_primal, "cutoff", cutoff, "precision", precision, "xshape", x_primal.shape)

                        self.compare_forces(
                            x_primal,
                            lamb_primal,
                            x_tangent,
                            lamb_tangent,
                            ref_nb_parameterized,
                            test_force,
                            precision,
                            rtol=rtol
                        )

                        primals = (x_primal, lamb_primal, charge_params, lj_params)
                        tangents = (x_tangent, lamb_tangent, np.zeros_like(charge_params), np.zeros_like(lj_params))

                        grad_fn = jax.grad(ref_nb, argnums=(0, 1, 2, 3))
                        ref_primals, ref_tangents = jax.jvp(grad_fn, primals, tangents)

                        ref_du_dcharge_primals = ref_primals[2]
                        test_du_dcharge_primals = test_force.get_du_dcharge_primals()
                        np.testing.assert_almost_equal(ref_du_dcharge_primals, test_du_dcharge_primals, rtol)

                        ref_du_dcharge_tangents = ref_tangents[2]
                        test_du_dcharge_tangents = test_force.get_du_dcharge_tangents()
                        np.testing.assert_almost_equal(ref_du_dcharge_tangents, test_du_dcharge_tangents, rtol)

                        ref_du_dlj_primals = ref_primals[3]
                        test_du_dlj_primals = test_force.get_du_dlj_primals()
                        np.testing.assert_almost_equal(ref_du_dlj_primals, test_du_dlj_primals, rtol)

                        ref_du_dlj_tangents = ref_tangents[3]
                        test_du_dlj_tangents = test_force.get_du_dlj_tangents()
                        np.testing.assert_almost_equal(ref_du_dlj_tangents, test_du_dlj_tangents, rtol)


if __name__ == "__main__":
    unittest.main()
