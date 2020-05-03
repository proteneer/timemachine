# test linear mixing
import unittest
import numpy as np

from fe.linear_mixer import LinearMixer

class TestLinearMixer(unittest.TestCase):

    def test_mix_exclusions(self):

        map_a_to_b = {
            0: 0,
            1: 1,
            2: 2
        }

        n_a = 6

        lm = LinearMixer(n_a, map_a_to_b)

        exc_a = np.array([
            [0,1],
            [0,2],
            [1,2],
            [2,3],
            [3,4],
            [4,5]
        ], dtype=np.int32)

        exc_a_params = np.array([
            0.1,
            0.2,
            1.2,
            2.3,
            3.4,
            4.5,
        ])

        exc_b = np.array([
            [0,1],
            [1,2],
            [0,2],
            [2,3]
        ], dtype=np.int32)

        exc_b_params = np.array([
            6.7,
            7.8,
            6.8,
            8.9
        ])


        (lhs_exclusions, lhs_param_exclusions), (rhs_exclusions, rhs_param_exclusions) = lm.mix_exclusions(
            exc_a,
            exc_a_params,
            exc_b,
            exc_b_params)

        lhs = {}
        for k, v in zip(lhs_exclusions, lhs_param_exclusions):
            lhs[k] = v

        assert lhs == {
            (0,1) : 0.1,
            (0,2) : 0.2,
            (1,2) : 1.2,
            (2,3) : 2.3,
            (3,4) : 3.4,
            (4,5) : 4.5,
            (6,7) : 6.7,
            (6,8) : 6.8,
            (7,8) : 7.8,
            (8,9) : 8.9,
            (2,9) : 8.9,  
            (3,8) : 2.3, # rhs exclusions
        }

        rhs = {}
        for k, v in zip(rhs_exclusions, rhs_param_exclusions):
            rhs[k] = v

        assert rhs == {
            (0,2) : 6.8,
            (1,2) : 7.8,
            (0,1) : 6.7,
            (2,9) : 8.9,
            (6,7) : 0.1,
            (7,8) : 1.2,
            (6,8) : 0.2,
            (3,8) : 2.3,
            (3,4) : 3.4,
            (4,5) : 4.5,
            (2,3) : 2.3,
            (8,9) : 8.9,
        }

    def test_mix_nonbonded(self):

        map_a_to_b = {
            0: 4,
            1: 0,
            2: 1
        }

        n_a = 5
        n_b = 6


        lm = LinearMixer(n_a, map_a_to_b)

        lambda_plane_idxs, lambda_offset_idxs = lm.mix_lambda_planes(n_a, n_b)

        np.testing.assert_equal(lambda_plane_idxs,  [1,1,1,1,1,0,0,0,0,0,0])
        np.testing.assert_equal(lambda_offset_idxs, [0,0,0,1,1,0,0,1,1,0,1])

    def test_mix_nonbonded_parameters(self):
        n_a = 5
        map_a_to_b = {
            0: 4,
            1: 0,
            2: 1
        }
        lm = LinearMixer(n_a, map_a_to_b)
        #                    0 1 2 3 4
        params_a = np.array([1,5,3,2,0])
        #                    5 6 7 8 9
        params_b = np.array([6,5,2,2,3])
        lhs_params, rhs_params = lm.mix_nonbonded_parameters(params_a, params_b)
        np.testing.assert_equal(lhs_params, np.concatenate([params_a, params_b]))

        #                        0 1 2 3 4 
        new_params_a = np.array([3,6,5,2,0])
        #                        
        new_params_b = np.array([5,3,2,2,1])
        np.testing.assert_equal(rhs_params, np.concatenate([new_params_a, new_params_b]))

    def test_mix_bonds(self):

        n_a = 5
        map_a_to_b = {
            0: 4,
            1: 0,
            2: 1
        }

        lm = LinearMixer(n_a, map_a_to_b)

        a_bond_idxs = np.array([
            [2, 3],
            [0, 1],
            [1, 2],
            [0, 2],
            [1, 4]
        ])

        a_param_idxs = np.array([
            'a0',
            'a1',
            'a2',
            'a3',
            'a4'
        ])

        b_bond_idxs = np.array([
            [0, 4],
            [2, 3],
            [0, 1],
            [1, 4],
            [1, 3]
        ])

        b_param_idxs = np.array([
            'b0',
            'b1',
            'b2',
            'b3',
            'b4'
        ])

        lhs_bond_idxs, lhs_param_idxs, rhs_bond_idxs, rhs_param_idxs = lm.mix_arbitrary_bonds(a_bond_idxs, a_param_idxs, b_bond_idxs, b_param_idxs)

        np.testing.assert_equal(lhs_bond_idxs, np.concatenate([a_bond_idxs, b_bond_idxs+n_a]))
        np.testing.assert_equal(lhs_param_idxs, np.concatenate([a_param_idxs, b_param_idxs]))

        rhs_part_one = np.array([
            [1, 0],
            [7, 8],
            [1, 2],
            [2, 0],
            [2, 8]
        ])

        rhs_part_two = np.array([
            [6, 3],
            [9, 5],
            [5, 6],
            [9, 6],
            [5, 4]
        ])

        np.testing.assert_equal(rhs_bond_idxs, np.concatenate([rhs_part_one, rhs_part_two]))
        np.testing.assert_equal(rhs_param_idxs, np.concatenate([b_param_idxs, a_param_idxs]))
