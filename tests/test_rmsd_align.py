# test the custom_op for rmsd alignment

from timemachine.lib import custom_ops

from timemachine.potentials import rmsd
import numpy as np
from testsystems import relative
from scipy.stats import special_ortho_group


def test_rmsd_align():

    N = 25

    for _ in range(100):

        x1 = np.random.rand(N,3)
        random_t = np.random.rand(3)
        random_R = special_ortho_group.rvs(3)
        x2 = x1@random_R + random_t

        assert np.linalg.norm(x2 - x1) > 1e-6

        x2_aligned_reference = rmsd.align_x2_unto_x1(x1, x2)

        # com2 should equal to that of com1
        np.testing.assert_almost_equal(np.mean(x1, axis=0), np.mean(x2_aligned_reference, axis=0))

        assert np.linalg.norm(x2_aligned_reference - x1) < 1e-6

        x2_aligned_test = custom_ops.rmsd_align(x1, x2)

        np.testing.assert_almost_equal(np.mean(x1, axis=0), np.mean(x2_aligned_test, axis=0))

        assert np.linalg.norm(x2_aligned_test - x1) < 1e-6

        np.testing.assert_almost_equal(x2_aligned_reference, x2_aligned_test)

