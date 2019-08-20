import unittest

import numpy as np
import jax
from jax.config import config; config.update("jax_enable_x64", True)
import functools

from timemachine.lib import custom_ops
from timemachine.potentials import bonded
from timemachine.potentials import nonbonded

def generate_derivatives(energy_fn, confs, params):
    E_fn = energy_fn
    dE_dx_fn = jax.grad(energy_fn, argnums=(0,))
    d2E_dx2_fn = jax.jacfwd(dE_dx_fn, argnums=(0,))
    dE_dp_fn = jax.grad(energy_fn, argnums=(1,))
    d2E_dxdp_fn = jax.jacfwd(dE_dp_fn, argnums=(0,))

    a = []
    b = []
    c = []
    d = []
    e = []
    for conf in confs:
        a.append(E_fn(conf, params))
        b.append(dE_dx_fn(conf, params)[0])
        c.append(d2E_dx2_fn(conf, params)[0][0])
        d.append(dE_dp_fn(conf, params)[0])
        e.append(d2E_dxdp_fn(conf, params)[0][0])

    a = np.stack(a)
    b = np.stack(b)
    c = np.stack(c)
    d = np.stack(d)
    e = np.stack(e)

    return a,b,c,d,e


class CustomOpsTest(unittest.TestCase):

    def assert_derivatives(self, conf, params, ref_nrg, test_nrg):

        ndims = conf.shape[-1]

        if conf.ndim == 2:
            num_confs = np.random.randint(1, 10)
            confs = np.repeat(conf[np.newaxis, :, :], num_confs, axis=0)
            # confs += np.random.rand(*confs.shape)
        else:
            confs = conf
            num_confs = confs.shape[0]

        num_atoms = confs.shape[1]

        all_dp_idxs = [
            np.array([]),
            np.random.permutation(np.arange(len(params)))[:np.random.randint(len(params))],
            np.arange(len(params))
        ]

        for dp_idxs in all_dp_idxs:

            dp_idxs = dp_idxs.astype(np.int32)
            test_e, test_de_dx, test_d2e_dx2, test_de_dp, test_d2e_dxdp = test_nrg.derivatives(
                confs,
                params,
                dp_idxs=dp_idxs
            )

            ref_e, ref_de_dx, ref_d2e_dx2, ref_de_dp, ref_d2e_dxdp = generate_derivatives(ref_nrg, confs, params)

            # batch compare
            np.testing.assert_almost_equal(test_e, ref_e)
            np.testing.assert_almost_equal(test_de_dx, ref_de_dx)

            def symmetrize(a):
                a = a.reshape(num_atoms*ndims, num_atoms*ndims)
                a = np.tril(a)
                a = a + a.T - np.diag(a.diagonal())
                return a.reshape(num_atoms, ndims, num_atoms, ndims)

            # test symmetric hessians
            for t, r in zip(test_d2e_dx2, ref_d2e_dx2):

                t = symmetrize(t)
                # np.testing.assert_almost_equal(t, r)

                # symmetrize()

                # t = t[:, :3, :, :3]
                # r = r[:, :3, :, :3]
                # test_tril = np.tril(np.reshape(t, (num_atoms*4, num_atoms*4)))
                # ref_tril = np.tril(np.reshape(r, (num_atoms*4, num_atoms*4)))
                # print("REF TRIL", ref_tril)
                # print("TEST_TRIL", test_tril)
                print(t[:, 3:, :, 3:])
                print(r[:, 3:, :, 3:])
                np.testing.assert_almost_equal(t, r)

                # np.testing.assert_almost_equal(test_tril[:, 3:, :, 3:], ref_tril[:, :3, :, :3])

            # batch compare
            np.testing.assert_almost_equal(test_de_dp, ref_de_dp[:, dp_idxs])
            np.testing.assert_almost_equal(test_d2e_dxdp, ref_d2e_dxdp[:, dp_idxs, :, :])

    def assert_derivatives_mixed_dimensions(self, conf4d, params, ref_nrg, test_nrg):

        ndims = conf4d.shape[-1]

        if conf4d.ndim == 2:
            num_conf4ds = np.random.randint(1, 10)
            conf4ds = np.repeat(conf4d[np.newaxis, :, :], num_conf4ds, axis=0)
            conf4ds += np.random.rand(*conf4ds.shape)
        else:
            conf4ds = conf4d
            num_conf4ds = conf4ds.shape[0]

        num_atoms = conf4ds.shape[1]

        all_dp_idxs = [
            np.array([]),
            np.random.permutation(np.arange(len(params)))[:np.random.randint(len(params))],
            np.arange(len(params))
        ]

        for dp_idxs in all_dp_idxs:

            dp_idxs = dp_idxs.astype(np.int32)
            test_e, test_de_dx, test_d2e_dx2, test_de_dp, test_d2e_dxdp = test_nrg.derivatives(
                conf4ds,
                params,
                dp_idxs=dp_idxs
            )

            conf3ds = conf4ds[:, :, :3]

            ref_e, ref_de_dx, ref_d2e_dx2, ref_de_dp, ref_d2e_dxdp = generate_derivatives(ref_nrg, conf3ds, params)

            # batch compare
            np.testing.assert_almost_equal(test_e, ref_e)
            np.testing.assert_almost_equal(test_de_dx[:, :, :3], ref_de_dx)
            np.testing.assert_almost_equal(test_de_dx[:, :, -1], np.zeros_like(test_de_dx[:, :, -1]))

            # test symmetric hessians
            for t, r in zip(test_d2e_dx2, ref_d2e_dx2):
                t = t[:, :3, :, :3]
                test_tril = np.tril(np.reshape(t, (num_atoms*3, num_atoms*3)))
                ref_tril = np.tril(np.reshape(r, (num_atoms*3, num_atoms*3)))
                np.testing.assert_almost_equal(test_tril, ref_tril)

            # batch compare
            np.testing.assert_almost_equal(test_de_dp, ref_de_dp[:, dp_idxs])
            # [C, P, N, 3]
            np.testing.assert_almost_equal(test_d2e_dxdp[:, :, :, :3], ref_d2e_dxdp[:, dp_idxs, :, :])


class TestHarmonicBond(CustomOpsTest):

    def test_derivatives(self):

        x0 = np.array([
            [1.0, 0.2, 3.3], # H 
            [-0.5,-1.1,-0.9], # C
            [3.4, 5.5, 0.2], # H 
            [3.2, 5.6, 0.5], # H 
        ], dtype=np.float64)

        params = np.array([10.0, 3.0, 1.5], dtype=np.float64)
        param_idxs = np.array([
            [2,1],
            [2,0],
            [0,1],
        ], dtype=np.int32)

        bond_idxs = np.array([
            [0,1],
            [1,2],
            [2,3]
        ], dtype=np.int32)

        hb = custom_ops.HarmonicBond_f64(
            bond_idxs,
            param_idxs
        )

        energy_fn = functools.partial(
            bonded.harmonic_bond,
            box=None,
            param_idxs=param_idxs,
            bond_idxs=bond_idxs
        )

        self.assert_derivatives(
            x0,
            params,
            energy_fn,
            hb
        )

        # test a specialized variant of 4D derivatives truncating to 3D
        x0_4d = np.array([
            [1.0,  0.2, 3.3,  0], # H 
            [-0.5,-1.1,-0.9,  0], # C
            [3.4,  5.5, 0.2,  0], # H 
            [3.2,  5.6, 0.5,  0], # H 
        ], dtype=np.float64)

        self.assert_derivatives_mixed_dimensions(
            x0_4d,
            params,
            energy_fn,
            hb
        )

    def test_special_derivatives(self):
        x0 = np.array([
            [  2.69220064,   1.97004635,  -2.03574268,   0.     ],
            [  2.50147346,   0.79925341,  -0.43194355,   0.     ],
            [  0.31155795,   0.76004772,   0.68364305,   0.     ],
            [-10.4245315 ,  -4.80327719,  -0.30447288,   0.1    ],
            [  0.7704191 ,  -0.41884917,   3.01832176,   0.1    ],
        ], dtype=np.float64)

        x0.setflags(write=False)

        params = np.array([100.0, 2.0, 75.0, 1.81, 3.0, 2.0, 1.0, 1.4], np.float64)
        bond_idxs = np.array([[0, 1], [1, 2], [3,4]], dtype=np.int32)
        bond_param_idxs = np.array([[0, 1], [0, 1], [0,1]], dtype=np.int32)

        energy_fn = functools.partial(
            bonded.harmonic_bond,
            box=None,
            param_idxs=bond_param_idxs,
            bond_idxs=bond_idxs
        )

        hb = custom_ops.HarmonicBond_f64(
            bond_idxs,
            bond_param_idxs
        )

        self.assert_derivatives_mixed_dimensions(
            x0,
            params,
            energy_fn,
            hb
        )

class TestHarmonicAngle(CustomOpsTest):
    
    def test_derivatives(self):

        x0 = np.array([
            [ 0.0637,   0.0126,   0.2203], # C
            [ 1.0573,  -0.2011,   1.2864], # H
            [ 2.3928,   1.2209,  -0.2230], # H
            [-0.6891,   1.6983,   0.0780], # H
            [-0.6312,  -1.6261,  -0.2601], # H
        ], dtype=np.float64)
        num_atoms = x0.shape[0]
        params = np.array([75, 1.91, 0.45], dtype=np.float64)

        angle_idxs = np.array([[1,0,2],[1,0,3],[1,0,4],[2,0,3],[2,0,4],[3,0,4]], dtype=np.int32)
        param_idxs = np.array([[0,1],[0,1],[0,2],[0,1],[0,1],[0,2]], dtype=np.int32)

        # enable cos angles
        energy_fn = functools.partial(
            bonded.harmonic_angle,
            box=None,
            angle_idxs=angle_idxs,
            param_idxs=param_idxs,
            cos_angles=True)

        ha = custom_ops.HarmonicAngle_f64(
            angle_idxs,
            param_idxs
        )

        self.assert_derivatives(
            x0,
            params,
            energy_fn,
            ha
        )

        # test a specialized variant of 4D derivatives truncating to 3D
        x0_4d = np.array([
            [ 0.0637,   0.0126,   0.2203, 0.1], # C
            [ 1.0573,  -0.2011,   1.2864, 0.2], # H
            [ 2.3928,   1.2209,  -0.2230, 0.3], # H
            [-0.6891,   1.6983,   0.0780, 0.4], # H
            [-0.6312,  -1.6261,  -0.2601, 0.5], # H
        ], dtype=np.float64)

        self.assert_derivatives_mixed_dimensions(
            x0_4d,
            params,
            energy_fn,
            ha
        )


class TestPeriodicTorsion(CustomOpsTest):
    
    def test_derivatives(self):

        x0 = np.array([
            [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
             [ 0.561317027011325 , 0.2066950040043141, 0.3670430960815993],
             [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
             [ 0.9399773448903637,-0.6888774474110431, 0.2104211949995816]],
            [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
             [ 0.5613170270113252, 0.2066950040043142, 0.3670430960815993],
             [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
             [ 1.283345455745044 ,-0.0356257425880843,-0.2573923896494185]],
            [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
             [ 0.561317027011325 , 0.2066950040043142, 0.3670430960815992],
             [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
             [ 1.263820400176392 , 0.7964992122869241, 0.0084568741589791]],
            [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
             [ 0.5613170270113252, 0.2066950040043142, 0.3670430960815992],
             [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
             [ 0.8993534242298198, 1.042445571242743 , 0.7635483993060286]],
            [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
             [ 0.5613170270113255, 0.2066950040043142, 0.3670430960815993],
             [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
             [ 0.5250337847650304, 0.476091386095139 , 1.3136545198545133]],
            [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
             [ 0.5613170270113255, 0.2066950040043141, 0.3670430960815993],
             [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
             [ 0.485009232042489 ,-0.3818599172073237, 1.1530102055165103]],
            [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
             [ 0.5613170270113252, 0.2066950040043142, 0.3670430960815993],
             [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
             [ 1.2278668040866427, 0.8805184219394547, 0.099391329616366 ]],
            [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
             [ 0.561317027011325 , 0.206695004004314 , 0.3670430960815994],
             [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
             [ 0.5494071252089705,-0.5626592973923106, 0.9817919758125693]],

            ], dtype=np.float64)

        torsion_idxs = np.array([
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ], dtype=np.int32)

        params = np.array([
            2.3, # k0
            5.4, # k1
            9.0, # k2
            0.0, # t0
            3.0, # t1
            5.8, # t2
            1.0, # n0
            2.0, # n1
            3.0  # n2
        ])

        param_idxs = np.array([
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8]
        ], dtype=np.int32)

        energy_fn = functools.partial(
            bonded.periodic_torsion,
            param_idxs=param_idxs,
            torsion_idxs=torsion_idxs,
            box=None)

        pt = custom_ops.PeriodicTorsion_f64(
            torsion_idxs,
            param_idxs
        )

        # (ytz): keep sanity snippet
        # from simtk import openmm

        # sys = openmm.System()
        # force = openmm.PeriodicTorsionForce()
        # force.addTorsion(
        #     0, 1, 2, 3,
        #     1, 0.5, 2.3
        # )
        # sys.addForce(force)
        # sys.addParticle(1.0)
        # sys.addParticle(1.0)
        # sys.addParticle(1.0)
        # sys.addParticle(1.0)

        # ctxt = openmm.Context(sys, openmm.LangevinIntegrator(1.0, 1.0, 1.0))

        # ctxt.setPositions(x0)
        # ctxt.setPeriodicBoxVectors(
        #     [10.0,  0.0,  0.0],
        #     [ 0.0, 10.0,  0.0],
        #     [ 0.0,  0.0, 10.0]
        # )

        # s = ctxt.getState(getEnergy=True, getForces=True)
        # print("OpenMM energy:", s.getPotentialEnergy())
        # for f in s.getForces():
        #     print("OpenMM forces", f)


        self.assert_derivatives(
            x0,
            params,
            energy_fn,
            pt
        )

        x0_4d = np.zeros(shape=(8, 4, 4))
        x0_4d[:,:,:3] = x0

        self.assert_derivatives(
            x0_4d,
            params,
            energy_fn,
            pt
        )  


class TestLennardJones(CustomOpsTest):

    def test_derivatives_special(self):

        x0 = np.array([
            [  2.69220064,   1.97004635,  -2.03574268,   0.     ],
            [  2.50147346,   0.79925341,  -0.43194355,   0.     ],
            [  0.31155795,   0.76004772,   0.68364305,   0.     ],
            [-10.4245315 ,  -4.80327719,  -0.30447288,   0.1    ],
            [  0.7704191 ,  -0.41884917,   3.01832176,   0.1    ],
        ], dtype=np.float64)


        params = np.array([3.0, 2.0, 1.0, 1.4], dtype=np.float64)
        param_idxs = np.array([
            [0, 3],
            [1, 2],
            [1, 2],
            [1, 2],
            [0, 3]], dtype=np.int32)

        scale_matrix = np.array([
            [  0,  0,   0, 0.5, 0.5],
            [  0,  0,   0,   1,   1],
            [  0,  0,   0, 0.5, 0.5],
            [0.5,  1, 0.5,   0, 0.5],
            [0.5,  1, 0.5, 0.5,   0]
        ], dtype=np.float64)

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

        conf4d = np.array([
            [ 0.0637,   0.0126,   0.2203,  0.5],
            [ 1.0573,  -0.2011,   1.2864, -0.2],
            [ 2.3928,   1.2209,  -0.2230,  5.6],
            [-0.6891,   1.6983,   0.0780,  2.3],
            [-0.6312,  -1.6261,  -0.2601, -5.1]
        ], dtype=np.float64)

        self.assert_derivatives(
            conf4d,
            params,
            energy_fn,
            lj
        )

class TestElectrostatics(CustomOpsTest):

    def test_aperiodic_electrostatics(self):
        conf = np.array([
            [ 0.0637,   0.0126,   0.2203],
            [ 1.0573,  -0.2011,   1.2864],
            [ 2.3928,   1.2209,  -0.2230],
            [-0.6891,   1.6983,   0.0780],
            [-0.6312,  -1.6261,  -0.2601]
        ], dtype=np.float64)

        params = np.array([1.3, 0.3], dtype=np.float64)
        param_idxs = np.array([0, 1, 1, 1, 1], dtype=np.int32)
        scale_matrix = np.array([
            [  0,  1,  1,  1,0.5],
            [  1,  0,  0,  1,  1],
            [  1,  0,  0,  0,0.2],
            [  1,  1,  0,  0,  1],
            [0.5,  1,0.2,  1,  0],
        ], dtype=np.float64)

        # scale_matrix = np.array([
        #     [  0,  1,  1,  1,  1],
        #     [  1,  0,  0,  1,  1],
        #     [  1,  0,  0,  0,  1],
        #     [  1,  1,  0,  0,  1],
        #     [  1,  1,  1,  1,  0],
        # ], dtype=np.float64)

        # warning: non net-neutral cell
        # ref_nrg = nonbonded.electrostatic(param_idxs, scale_matrix)

        energy_fn = functools.partial(
            nonbonded.electrostatics,
            param_idxs=param_idxs,
            scale_matrix=scale_matrix,
            box=None)

        es = custom_ops.Electrostatics_f64(
            scale_matrix,
            param_idxs
        )

        self.assert_derivatives(
            conf,
            params,
            energy_fn,
            es
        )

        # test 4 dimensional hessians
        conf4d = np.array([
            [ 0.0637,   0.0126,   0.2203,  0.5],
            [ 1.0573,  -0.2011,   1.2864, -0.2],
            [ 2.3928,   1.2209,  -0.2230,  5.6],
            [-0.6891,   1.6983,   0.0780,  2.3],
            [-0.6312,  -1.6261,  -0.2601, -5.1]
        ], dtype=np.float64)

        self.assert_derivatives(
            conf4d,
            params,
            energy_fn,
            es
        )