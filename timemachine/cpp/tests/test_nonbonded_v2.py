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

from timemachine.lib import ops, custom_ops


np.set_printoptions(linewidth=500)

class TestNonbonded(GradientTest):

    def get_water_system(self,
        D,
        P_charges,
        P_lj,
        sort=False):

        x = np.load("water.npy").astype(np.float64)
        # x = x[:2976, :D]
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
        np.random.seed(125)
        N = 65
        D = 3
        E = 10
        P_charges = 4
        P_lj = 5
        P_exc = 7
 
        x = self.get_random_coords(N, D)

        # for precision, rtol in [(np.float64, 1e-9), (np.float32, 5e-6)]:
        for precision, rtol in [(np.float64, 1e-9)]:
            for cutoff in [50.0, 0.5, 0.3]:
                params, ref_forces, test_forces = prepare_nonbonded_system(
                    x,
                    E,
                    P_charges,
                    P_lj,
                    P_exc,
                    p_scale=10.0,
                    cutoff=cutoff,
                    precision=precision
                )

                for lamb in [0.0, cutoff/10,  cutoff/2, cutoff/1.2, cutoff]:

                    print("lambda", lamb, "cutoff", cutoff)
                    for r, t in zip(ref_forces, test_forces):
                        self.compare_forces(
                            x,
                            params,
                            lamb,
                            r,
                            t,
                            precision,
                            rtol=rtol
                        )

    def test_water_box(self):
        
        np.random.seed(123)
        P_charges = 4
        P_lj = 5
        P_exc = 7
        dim = 3

        for precision, rtol in [(np.float32, 2e-5), (np.float64, 5e-10)]:

            x = self.get_water_coords(dim)
            E = x.shape[0] # each water 2 bonds and 1 angle constraint, so we remove them.
            for cutoff in [1000.0, 0.9, 0.5, 0.001]:

                params, ref_forces, test_forces = prepare_nonbonded_system(
                    x,
                    E,
                    P_charges,
                    P_lj,
                    P_exc,
                    p_scale=10.0,
                    e_scale=0.5, # double the charges
                    cutoff=cutoff,
                    precision=precision
                )

                for lamb in [0.0, cutoff/10,  cutoff/2, cutoff/1.2, cutoff]:

                    for r, t in zip(ref_forces, test_forces):
                        self.compare_forces(
                            x,
                            params,
                            lamb,
                            r,
                            t,
                            precision,
                            rtol)

if __name__ == "__main__":
    unittest.main()
