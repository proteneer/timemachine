# test linear mixing
import unittest
import numpy as np

from fe.linear_mixer import LinearMixer

class TestLinearMixer(unittest.TestCase):

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

        lhs_bond_idxs, lhs_param_idxs, rhs_bond_idxs, rhs_param_idxs = lm.mix_bonds(a_bond_idxs, a_param_idxs, b_bond_idxs, b_param_idxs)

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