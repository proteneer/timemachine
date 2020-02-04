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

    # def prepare_system(self,
    #     x,
    #     P_bonds,
    #     P_angles,
    #     P_torsions,
    #     B,
    #     A,
    #     T):

    #     N = x.shape[0]
    #     D = x.shape[1]

    #     atom_idxs = np.arange(N)

    #     assert N % 32 == 0 # warp is 32

    #     params = np.array([], dtype=np.float64);
    #     bond_params = np.random.rand(P_bonds).astype(np.float64)
    #     bond_param_idxs = np.random.randint(low=0, high=P_bonds, size=(B,2), dtype=np.int32) + len(params)
    #     bond_idxs = []
    #     for _ in range(B):
    #         bond_idxs.append(np.random.choice(atom_idxs, size=2, replace=False))
    #     bond_idxs = np.array(bond_idxs, dtype=np.int32)
    #     params = np.concatenate([params, bond_params])

    #     params = np.array([], dtype=np.float64);
    #     angle_params = np.random.rand(P_angles).astype(np.float64)
    #     angle_param_idxs = np.random.randint(low=0, high=P_angles, size=(A,2), dtype=np.int32) + len(params)
    #     angle_idxs = []
    #     for _ in range(A):
    #         angle_idxs.append(np.random.choice(atom_idxs, size=3, replace=False))
    #     angle_idxs = np.array(angle_idxs, dtype=np.int32)
    #     params = np.concatenate([params, angle_params])

    #     params = np.array([], dtype=np.float64);
    #     torsion_params = np.random.rand(P_torsions).astype(np.float64)
    #     torsion_param_idxs = np.random.randint(low=0, high=P_torsions, size=(T,3), dtype=np.int32) + len(params)
    #     torsion_idxs = []
    #     for _ in range(T):
    #         torsion_idxs.append(np.random.choice(atom_idxs, size=4, replace=False))
    #     torsion_idxs = np.array(torsion_idxs, dtype=np.int32)

    #     params = np.concatenate([params, torsion_params])

    #     return x, params, bond_idxs, bond_param_idxs, angle_idxs, angle_param_idxs, torsion_idxs, torsion_param_idxs

    # def compare_system(self, x, params, bond_idxs, bond_param_idxs, angle_idxs, angle_param_idxs, torsion_idxs, torsion_param_idxs):

    #     N = x.shape[0]
    #     D = x.shape[1]
    #     P = params.shape[0]

    #     custom_bonded = ops.HarmonicBond(bond_idxs, bond_param_idxs, D)
    #     harmonic_bond_fn = functools.partial(bonded.harmonic_bond, box=None, bond_idxs=bond_idxs, param_idxs=bond_param_idxs)

    #     custom_angles = ops.HarmonicAngle(angle_idxs, angle_param_idxs, D)
    #     harmonic_angle_fn = functools.partial(bonded.harmonic_angle, box=None, angle_idxs=angle_idxs, param_idxs=angle_param_idxs)

    #     custom_torsions = ops.PeriodicTorsion(torsion_idxs, torsion_param_idxs, D)
    #     periodic_torsion_fn = functools.partial(bonded.periodic_torsion, box=None, torsion_idxs=torsion_idxs, param_idxs=torsion_param_idxs)

    #     self.compare_forces(
    #         x,
    #         params,
    #         custom_torsions,
    #         periodic_torsion_fn
    #     )

    #     self.compare_forces(
    #         x,
    #         params,
    #         custom_angles,
    #         harmonic_angle_fn
    #     )


    #     self.compare_forces(
    #         x,
    #         params,
    #         custom_bonded,
    #         harmonic_bond_fn
    #     )


    def test_bonded(self):
        np.random.seed(125)
        N = 64
        P_bonds = 4
        P_angles = 6
        P_torsions = 13
        B = 35
        A = 36
        T = 37

        for D in [3,4]:

            x = self.get_random_coords(N, D)

            params, ref_bonds, custom_bonds = prepare_bonded_system(
                x,
                P_bonds,
                P_angles,
                P_torsions,
                B,
                A,
                T
            )

            for r, t in zip(ref_bonds, custom_bonds):
                self.compare_forces(
                    x,
                    params,
                    r,
                    t
                )

            # args = self.prepare_system(
            #     x,
            #     P_bonds,
            #     P_angles,
            #     P_torsions,
            #     B,
            #     A,
            #     T
            # )

            # self.compare_system(*args)