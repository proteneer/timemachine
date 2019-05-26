import unittest

import numpy as np
import custom_ops

# (ytz): TBD test in both modes.
from jax.config import config; config.update("jax_enable_x64", True)
import functools

import jax
from timemachine.potentials import bonded
from timemachine.potentials import nonbonded

def batch_mult_jvp(grad_fn, x, p, dxdp):
    dpdp = np.eye(p.shape[0])
    def apply_one(dxdp_i, dpdp_i):
        return jax.jvp(
            grad_fn,
            (x, p),
            (dxdp_i, dpdp_i)
        )
    a, b = jax.vmap(apply_one)(dxdp, dpdp)
    return a[0], b

class CustomOpsTest(unittest.TestCase):

    def assert_derivatives(self, conf, params, ref_nrg, test_nrg):

        if conf.ndim == 2:
            num_confs = np.random.randint(1, 10)
            confs = np.repeat(conf[np.newaxis, :, :], num_confs, axis=0)
            confs += np.random.rand(*confs.shape)
        else:
            confs = conf

        # todo: perturb by a small amount

        # This is a messy unit test that tests for:
        # correctness of the 4 derivatives against the reference implementation
        # sparse scattered indices
        # using an empty null array to denote zero dx_dp
        all_dx_dps = [
            np.random.rand(confs.shape[0], params.shape[0], confs.shape[1], confs.shape[2]),
            None, # special case to check when we don't provide an explicit dx_dp
        ]

        all_dp_idxs = [
            np.array([]),
            np.random.permutation(np.arange(len(params)))[:np.random.randint(len(params))],
            np.arange(len(params))
        ]

        for batch_dx_dp in all_dx_dps:

            all_ref_es = []
            all_ref_de_dps = []
            all_ref_de_dxs = []
            all_ref_d2e_dxdps = []

            for conf_idx, conf in enumerate(confs):

                if batch_dx_dp is None:
                    test_dx_dp = np.zeros(shape=(params.shape[0], confs.shape[1], confs.shape[2]))
                else:
                    test_dx_dp = batch_dx_dp[conf_idx]

                ref_e, ref_de_dp = batch_mult_jvp(ref_nrg, conf, params, test_dx_dp)

                grad_fn = jax.grad(ref_nrg, argnums=(0,))
                ref_de_dx, ref_d2e_dxdp = batch_mult_jvp(grad_fn, conf, params, test_dx_dp)
                
                all_ref_es.append(ref_e)
                all_ref_de_dps.append(ref_de_dp)
                all_ref_de_dxs.append(ref_de_dx[0])
                all_ref_d2e_dxdps.append(ref_d2e_dxdp[0])

            all_ref_es = np.stack(all_ref_es)
            all_ref_de_dps = np.stack(all_ref_de_dps)
            all_ref_de_dxs = np.stack(all_ref_de_dxs)
            all_ref_d2e_dxdps = np.stack(all_ref_d2e_dxdps)

            for dp_idxs in all_dp_idxs:
                dp_idxs = dp_idxs.astype(np.int32)

                if batch_dx_dp is None:
                    test_dx_dp = np.empty(shape=(0,))
                else:
                    test_dx_dp = batch_dx_dp[:, dp_idxs, :, :]


                test_e, test_de_dx, test_de_dp, test_d2e_dxdp = test_nrg.derivatives(
                    confs,
                    params,
                    dx_dp=test_dx_dp, # gather the ones we care about
                    dp_idxs=dp_idxs
                )

                np.testing.assert_almost_equal(test_e, all_ref_es)
                np.testing.assert_almost_equal(test_de_dp, all_ref_de_dps[:, dp_idxs]) # [C, P]
                np.testing.assert_almost_equal(test_de_dx, all_ref_de_dxs)
                np.testing.assert_almost_equal(test_d2e_dxdp, all_ref_d2e_dxdps[:, dp_idxs, :, :]) # [C, P, N, 3]


# class TestHarmonicBond(CustomOpsTest):

#     def test_derivatives(self):

#         x0 = np.array([
#             [1.0, 0.2, 3.3], # H 
#             [-0.5,-1.1,-0.9], # C
#             [3.4, 5.5, 0.2], # H 
#         ], dtype=np.float64)

#         params = np.array([10.0, 3.0, 5.5], dtype=np.float64)
#         param_idxs = np.array([
#             [0,1],
#             [1,2],
#         ], dtype=np.int32)

#         bond_idxs = np.array([
#             [0,1],
#             [1,2]
#         ], dtype=np.int32)

#         hb = custom_ops.HarmonicBond_f64(
#             bond_idxs,
#             param_idxs
#         )

#         energy_fn = functools.partial(
#             bonded.harmonic_bond,
#             box=None,
#             param_idxs=param_idxs,
#             bond_idxs=bond_idxs
#         )

#         self.assert_derivatives(
#             x0,
#             params,
#             energy_fn,
#             hb
#         )

# class TestHarmonicAngle(CustomOpsTest):
    
#     def test_derivatives(self):

#         x0 = np.array([
#             [ 0.0637,   0.0126,   0.2203], # C
#             [ 1.0573,  -0.2011,   1.2864], # H
#             [ 2.3928,   1.2209,  -0.2230], # H
#             [-0.6891,   1.6983,   0.0780], # H
#             [-0.6312,  -1.6261,  -0.2601], # H
#         ], dtype=np.float64)
#         num_atoms = x0.shape[0]
#         params = np.array([75, 1.91, 0.45], dtype=np.float64)

#         angle_idxs = np.array([[1,0,2],[1,0,3],[1,0,4],[2,0,3],[2,0,4],[3,0,4]], dtype=np.int32)
#         param_idxs = np.array([[0,1],[0,1],[0,2],[0,1],[0,1],[0,2]], dtype=np.int32)

#         # enable cos angles
#         energy_fn = functools.partial(
#             bonded.harmonic_angle,
#             box=None,
#             angle_idxs=angle_idxs,
#             param_idxs=param_idxs,
#             cos_angles=True)

#         ha = custom_ops.HarmonicAngle_f64(
#             angle_idxs,
#             param_idxs
#         )

#         self.assert_derivatives(
#             x0,
#             params,
#             energy_fn,
#             ha
#         )


# class TestPeriodicTorsion(CustomOpsTest):
    
#     def test_derivatives(self):

#         x0 = np.array([
#             [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
#              [ 0.561317027011325 , 0.2066950040043141, 0.3670430960815993],
#              [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
#              [ 0.9399773448903637,-0.6888774474110431, 0.2104211949995816]],
#             [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
#              [ 0.5613170270113252, 0.2066950040043142, 0.3670430960815993],
#              [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
#              [ 1.283345455745044 ,-0.0356257425880843,-0.2573923896494185]],
#             [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
#              [ 0.561317027011325 , 0.2066950040043142, 0.3670430960815992],
#              [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
#              [ 1.263820400176392 , 0.7964992122869241, 0.0084568741589791]],
#             [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
#              [ 0.5613170270113252, 0.2066950040043142, 0.3670430960815992],
#              [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
#              [ 0.8993534242298198, 1.042445571242743 , 0.7635483993060286]],
#             [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
#              [ 0.5613170270113255, 0.2066950040043142, 0.3670430960815993],
#              [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
#              [ 0.5250337847650304, 0.476091386095139 , 1.3136545198545133]],
#             [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
#              [ 0.5613170270113255, 0.2066950040043141, 0.3670430960815993],
#              [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
#              [ 0.485009232042489 ,-0.3818599172073237, 1.1530102055165103]],
#             [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
#              [ 0.5613170270113252, 0.2066950040043142, 0.3670430960815993],
#              [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
#              [ 1.2278668040866427, 0.8805184219394547, 0.099391329616366 ]],
#             [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
#              [ 0.561317027011325 , 0.206695004004314 , 0.3670430960815994],
#              [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
#              [ 0.5494071252089705,-0.5626592973923106, 0.9817919758125693]],

#             ], dtype=np.float64)

#         torsion_idxs = np.array([
#             [0, 1, 2, 3],
#             [0, 1, 2, 3],
#             [0, 1, 2, 3],
#         ], dtype=np.int32)

#         params = np.array([
#             2.3, # k0
#             5.4, # k1
#             9.0, # k2
#             0.0, # t0
#             3.0, # t1
#             5.8, # t2
#             1.0, # n0
#             2.0, # n1
#             3.0  # n2
#         ])

#         param_idxs = np.array([
#             [0, 3, 6],
#             [1, 4, 7],
#             [2, 5, 8]
#         ], dtype=np.int32)

#         energy_fn = functools.partial(
#             bonded.periodic_torsion,
#             param_idxs=param_idxs,
#             torsion_idxs=torsion_idxs,
#             box=None)

#         pt = custom_ops.PeriodicTorsion_f64(
#             torsion_idxs,
#             param_idxs
#         )

#         # (ytz): keep sanity snippet
#         # from simtk import openmm

#         # sys = openmm.System()
#         # force = openmm.PeriodicTorsionForce()
#         # force.addTorsion(
#         #     0, 1, 2, 3,
#         #     1, 0.5, 2.3
#         # )
#         # sys.addForce(force)
#         # sys.addParticle(1.0)
#         # sys.addParticle(1.0)
#         # sys.addParticle(1.0)
#         # sys.addParticle(1.0)

#         # ctxt = openmm.Context(sys, openmm.LangevinIntegrator(1.0, 1.0, 1.0))

#         # ctxt.setPositions(x0)
#         # ctxt.setPeriodicBoxVectors(
#         #     [10.0,  0.0,  0.0],
#         #     [ 0.0, 10.0,  0.0],
#         #     [ 0.0,  0.0, 10.0]
#         # )

#         # s = ctxt.getState(getEnergy=True, getForces=True)
#         # print("OpenMM energy:", s.getPotentialEnergy())
#         # for f in s.getForces():
#         #     print("OpenMM forces", f)

#         self.assert_derivatives(
#             x0,
#             params,
#             energy_fn,
#             pt
#         )

class TestLennardJones(CustomOpsTest):

    def test_derivatives(self):

        x0 = np.array([
            [ 0.0637,   0.0126,   0.2203],
            [ 1.0573,  -0.2011,   1.2864],
            [ 2.3928,   1.2209,  -0.2230],
            [-0.6891,   1.6983,   0.0780],
            [-0.6312,  -1.6261,  -0.2601]
        ], dtype=np.float64)

        params = np.array([3.0, 2.0, 1.0, 1.4], dtype=np.float64)
        param_idxs = np.array([
            [0, 3],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2]], dtype=np.int32)

        scale_matrix = np.array([
            [  0,  0,  1,0.5,  0],
            [  0,  0,  0,  1,  1],
            [  1,  0,  0,  0,0.2],
            [0.5,  1,  0,  0,  1],
            [  0,  1,0.2,  1,  0],
        ], dtype=np.float64)

        # box = np.array([
        #     [2.0, 0.5, 0.6],
        #     [0.6, 1.6, 0.3],
        #     [0.4, 0.7, 1.1]
        # ], dtype=np.float64)

        energy_fn = functools.partial(nonbonded.lennard_jones,
            scale_matrix=scale_matrix,
            param_idxs=param_idxs,
            box=None,
            cutoff=None)

        lj = custom_ops.LennardJones_f64(
            scale_matrix,
            param_idxs
        )

        self.assert_derivatives(
            x0,
            params,
            energy_fn,
            lj
        )