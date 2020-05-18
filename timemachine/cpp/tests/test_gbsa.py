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

from timemachine.potentials import alchemy
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

    def test_gbsa(self):

        D = 3
        np.random.seed(1523)

        x = self.get_water_coords(D, sort=True)
        N = x.shape[0]

        P_charges = N
        P_radii = N
        P_scale_factors = N

        dielectric_offset = 0.009
        solute_dielectric = 1.0
        solvent_dielectric = 78.5
 
        # for cutoff in [0.1, 1.0, 1.5, 2.0, 500.0]:
        # for cutoff in [50.0, 2.0, 1.0, 0.5, 0.1]:

        for l_idx in range(2):

            if l_idx == 0:

                # stage 1 and 3 use this
                lambda_plane_idxs = np.random.randint(
                    low=0,
                    high=1,
                    size=(N),
                    dtype=np.int32
                )

                lambda_offset_idxs = np.random.randint(
                    low=0,
                    high=2,
                    size=(N),
                    dtype=np.int32
                )
            elif l_idx == 1:

                # stage 2 has fully decoupled lambdas
                lambda_plane_idxs = np.random.randint(
                    low=0,
                    high=4,
                    size=(N),
                    dtype=np.int32
                )

                lambda_offset_idxs = np.random.randint(
                    low=0,
                    high=1,
                    size=(N),
                    dtype=np.int32
                )


            for cutoff in [10000.0]:
                print("Testing cutoff @", cutoff)
                for precision, rtol in [(np.float64, 1e-9), (np.float32, 8e-5)]:
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
                        precision=precision,
                        lambda_plane_idxs=lambda_plane_idxs,
                        lambda_offset_idxs=lambda_offset_idxs
                    )

                    for lamb in [0.0, 0.1,  0.5, 5.0]:


                        print(cutoff, lamb, precision)
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

    # def test_alchemical_gbsa(self):

    #     D = 3
    #     np.random.seed(1523)

    #     x = self.get_water_coords(D, sort=True)
    #     N = x.shape[0]

    #     P_charges = N
    #     P_radii = N
    #     P_scale_factors = N

    #     dielectric_offset = 0.009
    #     solute_dielectric = 1.0
    #     solvent_dielectric = 78.5
 
    #     # for cutoff in [0.1, 1.0, 1.5, 2.0, 500.0]:
    #     for cutoff in [50.0, 2.0, 1.0, 0.5, 0.1]:
    #         print("Testing cutoff @", cutoff)
    #         for precision, rtol in [(np.float64, 1e-9), (np.float32, 8e-5)]:
    #             params, ref_forces_0, test_forces_0 = prepare_gbsa_system(
    #                 x,
    #                 P_charges,
    #                 P_radii,
    #                 P_scale_factors,
    #                 alpha=0.35,
    #                 beta=0.645,
    #                 gamma=0.65,
    #                 dielectric_offset=dielectric_offset,
    #                 surface_tension=28.3919551,
    #                 solute_dielectric=solute_dielectric,
    #                 solvent_dielectric=solvent_dielectric,
    #                 probe_radius=0.14,
    #                 cutoff_radii=cutoff,
    #                 cutoff_force=cutoff,
    #                 precision=precision
    #             )

    #             params, ref_forces_1, test_forces_1 = prepare_gbsa_system(
    #                 x,
    #                 P_charges,
    #                 P_radii,
    #                 P_scale_factors,
    #                 alpha=0.35,
    #                 beta=0.645,
    #                 gamma=0.65,
    #                 dielectric_offset=dielectric_offset,
    #                 surface_tension=28.3919551,
    #                 solute_dielectric=solute_dielectric,
    #                 solvent_dielectric=solvent_dielectric,
    #                 probe_radius=0.14,
    #                 cutoff_radii=cutoff,
    #                 cutoff_force=cutoff,
    #                 precision=precision,
    #                 params=params
    #             )

    #             ref_fn = functools.partial(
    #                 alchemy.linear_rescale,
    #                 fn0 = ref_forces_0[0],
    #                 fn1 = ref_forces_1[0]
    #             )

    #             test_fn = ops.AlchemicalGradient(
    #                 N,
    #                 len(params),
    #                 test_forces_0[0],
    #                 test_forces_1[0]
    #             )


    #             for lamb in [0.0, 1/10,  1/2, 1/1.2, 1.0]:
    #                 self.compare_forces(
    #                     x,
    #                     params,
    #                     lamb,
    #                     ref_fn,
    #                     test_fn,
    #                     precision,
    #                     rtol=rtol
    #                 )
