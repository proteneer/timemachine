import unittest
import numpy as np
import functools

from jax.config import config; config.update("jax_enable_x64", True)
from jax.test_util import check_grads

from tests.invariances import assert_potential_invariance
from timemachine.potentials import bonded


class TestAngles(unittest.TestCase):

    def test_jax_harmonic_angle(self):
        x0 = np.array([
            [ 0.0637,   0.0126,   0.2203], # C
            [ 1.0573,  -0.2011,   1.2864], # H
            [ 2.3928,   1.2209,  -0.2230], # H
            [-0.6891,   1.6983,   0.0780], # H
            [-0.6312,  -1.6261,  -0.2601], # H
        ], dtype=np.float64)
        num_atoms = x0.shape[0]
        params = np.array([75, 1.91, 0.45], dtype=np.float64)

        angle_idxs = np.array([[1,0,2],[1,0,3],[1,0,4],[2,0,3],[2,0,4],[3,0,4]])
        param_idxs = np.array([[0,1],[0,1],[0,2],[0,1],[0,1],[0,2]])

        # enable cos angles
        energy_fn = functools.partial(bonded.harmonic_angle,
            angle_idxs=angle_idxs,
            param_idxs=param_idxs,
            cos_angles=True)

        box = np.array([
            [2.0, 0.5, 0.6],
            [0.6, 1.6, 0.3],
            [0.4, 0.7, 1.1]
        ], dtype=np.float64)

        assert_potential_invariance(energy_fn, x0, params, box)

        # disable cos angles
        energy_fn = functools.partial(bonded.harmonic_angle,
            angle_idxs=angle_idxs,
            param_idxs=param_idxs,
            cos_angles=False)

        assert_potential_invariance(energy_fn, x0, params, box)


class TestBonded(unittest.TestCase):

    def test_jax_harmonic_bond(self):
        x0 = np.array([
            [1.0, 0.2, 3.3], # H 
            [-0.5,-1.1,-0.9], # C
            [3.4, 5.5, 0.2], # H 
        ], dtype=np.float64)

        params = np.array([10.0, 3.0, 5.5], dtype=np.float64)

        param_idxs = np.array([
            [0,1],
            [1,2],
        ], dtype=np.int32)

        bond_idxs = np.array([
            [0,1],
            [1,2]
        ], dtype=np.int32)

        energy_fn = functools.partial(bonded.harmonic_bond, param_idxs=param_idxs, bond_idxs=bond_idxs)

        box = np.array([
            [2.0, 0.5, 0.6],
            [0.6, 1.6, 0.3],
            [0.4, 0.7, 1.1]
        ], dtype=np.float64)

        assert_potential_invariance(energy_fn, x0, params, box)


class TestPeriodicTorsion(unittest.TestCase):

    def setUp(self):
        self.conformers = np.array([
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
            ], dtype=np.float64)

        self.nan_conformers = np.array([
            [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
             [ 0.5613170270113252, 0.2066950040043142, 0.3670430960815993],
             [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
             [ 1.2278668040866427, 0.8805184219394547, 0.099391329616366 ]],
            [[-0.6000563454193615, 0.376172954382274 ,-0.2487295756125901],
             [ 0.561317027011325 , 0.206695004004314 , 0.3670430960815994],
             [-1.187055522272264 ,-0.3415864358441354, 0.0871382207830652],
             [ 0.5494071252089705,-0.5626592973923106, 0.9817919758125693]],
            ], dtype=np.float64)


    def test_jax_torsions(self):
        """
        Test agreement of torsions with OpenMM's implementation of torsion terms.
        """
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

        box = np.array([
            [2.0, 0.5, 0.6],
            [0.6, 1.6, 0.3],
            [0.4, 0.7, 1.1]
        ], dtype=np.float64)

        energy_fn = functools.partial(
            bonded.periodic_torsion,
            param_idxs=param_idxs,
            torsion_idxs=torsion_idxs)

        # there's no good finite difference tests that we can do for the nan_conformers
        # so instead we compare against OpenMM implementation later on
        for conf_idx, conf in enumerate(self.conformers):
            assert_potential_invariance(energy_fn, conf, params, box)

if __name__ == "__main__":
    unittest.main()


