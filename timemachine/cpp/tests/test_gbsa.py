import functools
import unittest
import scipy.linalg
from jax.config import config; config.update("jax_enable_x64", True)

import numpy as np
import jax
import jax.numpy as jnp

import functools

from common import GradientTest
from common import prepare_gbsa_system

from timemachine.lib import custom_ops


class TestGBSA(GradientTest):

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

    def test_gbsa(self):

        D = 4
        np.random.seed(125)

        x = self.get_water_coords(D)
        N = x.shape[0]

        P_charges = 64
        P_radii = 12
        P_scale_factors = 12

        dielectric_offset = 0.009
        solute_dielectric = 1.0
        solvent_dielectric = 78.5
 
        cutoff = 1000.0
        for precision, rtol in [(np.float64, 1e-10), (np.float32, 8e-5)]:
            for cutoff in [10000.0]:
                np.random.rand(1)

                params, ref_forces, test_forces = prepare_gbsa_system(
                    x,
                    P_charges,
                    P_radii,
                    P_scale_factors,
                    alpha=0.35,
                    beta=0.645,
                    gamma=0.65,
                    dielectric_offset=dielectric_offset,
                    surface_tension=28.3919551,
                    solute_dielectric=solute_dielectric,
                    solvent_dielectric=solvent_dielectric,
                    probe_radius=0.14,
                    precision=precision
                )

                for r, t in zip(ref_forces, test_forces):
                    self.compare_forces(
                        x,
                        params,
                        r,
                        t,
                        precision,
                        rtol=rtol
                    )
