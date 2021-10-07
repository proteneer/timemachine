import unittest

import numpy as np

from training.dataset import Dataset

from testsystems.relative import hif2a_ligand_pair


class TestDataset(unittest.TestCase):

    def test_split(self):
        ds = Dataset(list(range(100)))
        count = len(ds)
        invalid_fracs = [-0.1, 1.1, 1.01, -1.0]
        valid_fracs = [0.1 * i for i in range(11)]

        # Verify that if fractions are out of bound, an exception is raised
        for frac in invalid_fracs:
            with self.assertRaises(ValueError):
                ds.split(frac)

        for frac in valid_fracs:
            lhs, rhs = ds.split(frac)
            self.assertEqual(len(lhs) + len(rhs), count)
            self.assertAlmostEqual(len(lhs) / count, frac)
            self.assertAlmostEqual(len(rhs) / count, 1.0 - frac)
            if len(lhs) and len(rhs):
                self.assertEqual(lhs.data[0], ds.data[0])
                self.assertEqual(rhs.data[0], ds.data[len(lhs)])

    def test_random_split(self):
        ds = Dataset(list(range(100)))
        count = len(ds)
        # Seed it so that checks on randomness don't break occasionally
        np.random.seed(814)
        valid_fracs = [0.1 * i for i in range(11)]

        for frac in valid_fracs:
            lhs, rhs = ds.random_split(frac)
            self.assertEqual(len(lhs) + len(rhs), count)
            self.assertAlmostEqual(len(lhs) / count, frac)
            self.assertAlmostEqual(len(rhs) / count, 1.0 - frac)
            if len(lhs) and len(rhs):
                self.assertNotEqual(lhs.data, ds.data[:len(lhs)])
                self.assertNotEqual(rhs.data[0], ds.data[len(lhs)])
        # Verify that original dataset didn't get shuffled in the process
        for i in range(count):
            self.assertEqual(ds.data[i], i)


    def test_indices_split(self):
        indices = list(range(100))
        ds = Dataset(indices.copy())

        # If indices are duplicated should see exception
        with self.assertRaises(ValueError):
            ds.indices_split(indices, indices)

        split_point = 25

        left = indices[:split_point]
        right = indices[split_point:]

        bad_indices = list(range(100, 200))
        bad_left = bad_indices[:split_point]
        bad_right = bad_indices[split_point:]
        # If invalid indices, should raise exception
        with self.assertRaises(ValueError):
            ds.indices_split(bad_left, bad_right)

        # If len(left + right) != len(ds), raise exception
        with self.assertRaises(ValueError):
            ds.indices_split(left, right[:1])

        lhs, rhs = ds.indices_split(left, right)
        self.assertEqual(len(lhs), len(left))
        self.assertEqual(len(rhs), len(rhs))

    def test_dataset_data_is_a_copy(self):
        data = [hif2a_ligand_pair.mol_a, hif2a_ligand_pair.mol_b]
        ds = Dataset(data)
        data.pop(0)
        self.assertEqual(len(data), 1)
        self.assertEqual(len(ds.data), 2)
