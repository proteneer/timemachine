import numpy as np
import pytest

from timemachine.fe.plots import plot_forward_and_reverse_ddg, plot_forward_and_reverse_dg

# All plotting tests are to be tested without gpus
pytestmark = [pytest.mark.nogpu]


def test_forward_and_reverse_ddg_plot():
    rng = np.random.default_rng(2023)
    ukln_shape = (47, 2, 2, 2000)
    dummy_solv_ukln = rng.random(size=ukln_shape)
    dummy_complex_ukln = rng.random(size=ukln_shape)

    plot_forward_and_reverse_ddg(dummy_solv_ukln, dummy_complex_ukln)


def test_forward_and_reverse_ddg_plot_validation():
    rng = np.random.default_rng(2023)
    ukln_shape = (47, 2, 2, 2000)
    dummy_solv_ukln = rng.random(size=ukln_shape)
    dummy_complex_ukln = rng.random(size=ukln_shape)

    with pytest.raises(AssertionError, match="fewer samples than frames_per_step"):
        plot_forward_and_reverse_ddg(dummy_solv_ukln, dummy_complex_ukln, frames_per_step=ukln_shape[-1] + 1)
    # Verify that with different size arrays it fails
    with pytest.raises(AssertionError):
        plot_forward_and_reverse_ddg(dummy_solv_ukln, dummy_complex_ukln[0])


def test_forward_and_reverse_dg_plot():
    rng = np.random.default_rng(2023)
    ukln_shape = (47, 2, 2, 2000)
    dummy_ukln = rng.random(size=ukln_shape)

    plot_forward_and_reverse_dg(dummy_ukln)


def test_forward_and_reverse_dg_plot_validation():
    rng = np.random.default_rng(2023)
    ukln_shape = (47, 2, 2, 2000)
    dummy_ukln = rng.random(size=ukln_shape)

    with pytest.raises(AssertionError, match="fewer samples than frames_per_step"):
        plot_forward_and_reverse_dg(dummy_ukln, frames_per_step=ukln_shape[-1] + 1)
