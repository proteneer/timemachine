from fe import utils
import numpy as np


def test_sanitize_energies():

    full_us = np.array([[15000.0, -5081923.0, 1598, 1.5, -23.0], [-423581.0, np.nan, -238, 13.5, 23.0]])

    test_us = utils.sanitize_energies(full_us, lamb_idx=3, cutoff=10000)

    expected_us = np.array([[np.inf, np.inf, 1598, 1.5, -23.0], [np.inf, np.inf, -238, 13.5, 23.0]])

    np.testing.assert_array_equal(test_us, expected_us)


def test_extract_delta_Us_from_U_knk():

    K = 4
    N = 8

    U_knk = np.random.rand(K, N, K)

    test_delta_Us = utils.extract_delta_Us_from_U_knk(U_knk)

    expected_delta_Us = np.array(
        [
            # fwd                          # rev
            [U_knk[0, :, 1] - U_knk[0, :, 0], U_knk[1, :, 0] - U_knk[1, :, 1]],
            [U_knk[1, :, 2] - U_knk[1, :, 1], U_knk[2, :, 1] - U_knk[2, :, 2]],
            [U_knk[2, :, 3] - U_knk[2, :, 2], U_knk[3, :, 2] - U_knk[3, :, 3]],
        ]
    )

    np.testing.assert_almost_equal(expected_delta_Us, test_delta_Us)
