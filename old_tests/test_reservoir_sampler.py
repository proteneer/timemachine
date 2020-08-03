import numpy as np
import unittest

from timemachine.reservoir_sampler import ReservoirSampler

class TestReservoirSampler(unittest.TestCase):

    def test_reservoir_sampler(self):

        n = 1000
        k = 10

        def make_sequence():
            for i in range(n):
                yield i

        keep_counts = np.zeros(n, dtype=np.int64)

        for idx in range(1000):

            rs = ReservoirSampler(make_sequence(), k)

            all_items = []
            for item in rs.sample():
                all_items.append(item)

            np.testing.assert_array_equal(all_items, list(range(n)))

            for keep_idx in rs.R:
                keep_counts[keep_idx] += 1

        # note that testing mean is completely useless here
        std = np.std(keep_counts.astype(np.float64))

        assert std < 5


if __name__ == "__main__":
    unittest.main()