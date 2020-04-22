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

from timemachine.lib import ops, custom_ops


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

    def test_limits(self):


        D = 3
        np.random.seed(125)

        x = self.get_water_coords(D, sort=True)

        N = x.shape[0]

        P_charges = N
        P_radii = N
        P_scale_factors = N

        dielectric_offset = 0.009
        solute_dielectric = 1.0
        solvent_dielectric = 78.5

        alpha=0.35
        beta=0.645
        gamma=0.65

        surface_tension=28.3919551
        probe_radius=0.14

        cutoff_radii = 0.5
        cutoff_force = 0.5

        precision = np.float64

        params = np.array([], dtype=np.float64)

        # charges
        charge_params = (np.random.rand(P_charges).astype(np.float64)-0.5)*np.sqrt(138.935456)
        charge_param_idxs = np.random.randint(low=0, high=P_charges, size=(N), dtype=np.int32) + len(params)
        params = np.concatenate([params, charge_params])

        # gb radiis
        radii_params = 1.5*np.random.rand(P_radii).astype(np.float64) + 1.0 # 1.0 to 2.5
        radii_params = radii_params/10 # convert to nm form
        radii_param_idxs = np.random.randint(low=0, high=P_radii, size=(N), dtype=np.int32) + len(params)
        params = np.concatenate([params, radii_params])

        # scale factors
        scale_params = np.random.rand(P_scale_factors).astype(np.float64)/3 + 0.75
        scale_param_idxs = np.random.randint(low=0, high=P_scale_factors, size=(N), dtype=np.int32) + len(params)
        params = np.concatenate([params, scale_params])

        N_limits = N - 10

        custom_gb_full = ops.GBSA(
            charge_param_idxs,
            radii_param_idxs,
            scale_param_idxs,
            alpha,
            beta,
            gamma,
            dielectric_offset,
            surface_tension,
            solute_dielectric,
            solvent_dielectric,
            probe_radius,
            cutoff_radii,
            cutoff_force,
            N_limits,
            D,
            precision=precision
        )

        test_dx_full = custom_gb_full.execute(x, params)

        np.testing.assert_almost_equal(test_dx_full[N_limits:], 0)

        custom_gb_full_limit = ops.GBSA(
            charge_param_idxs[:N_limits],
            radii_param_idxs[:N_limits],
            scale_param_idxs[:N_limits],
            alpha,
            beta,
            gamma,
            dielectric_offset,
            surface_tension,
            solute_dielectric,
            solvent_dielectric,
            probe_radius,
            cutoff_radii,
            cutoff_force,
            N_limits,
            D,
            precision=precision
        )

        test_dx_limit = custom_gb_full.execute(x[:N_limits], params)

        np.testing.assert_almost_equal(test_dx_full[:N_limits], test_dx_limit)


    def test_gbsa(self):

        D = 3
        np.random.seed(125)

        # N = 33
        # x = self.get_random_coords(N, D) 

        x = self.get_water_coords(D, sort=True)

        N = x.shape[0]

        P_charges = N
        P_radii = N
        P_scale_factors = N

        dielectric_offset = 0.009
        solute_dielectric = 1.0
        solvent_dielectric = 78.5
 
        for cutoff in [0.1, 1.0, 1.5, 2.0, 500.0]:
            print("Testing cutoff @", cutoff)
            for precision, rtol in [(np.float32, 8e-5), (np.float64, 1e-10)]:
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
                    cutoff_radii=cutoff,
                    cutoff_force=cutoff,
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
