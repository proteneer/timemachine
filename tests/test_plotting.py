import numpy as np
import pytest

from timemachine.fe.plots import plot_forward_and_reverse_ddg, plot_forward_and_reverse_dg

# Plotting code should not depend on CUDA
pytestmark = [pytest.mark.nocuda]


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


@pytest.mark.parametrize("ukln_shape", [(47, 2, 2, 2000), (5, 2, 2, 10)])
def test_forward_and_reverse_dg_plot(ukln_shape):
    rng = np.random.default_rng(2023)
    dummy_ukln = rng.random(size=ukln_shape) * 1000

    frames_per_step = min(ukln_shape[-1], 100)
    plot_forward_and_reverse_dg(dummy_ukln, frames_per_step=frames_per_step)


def test_forward_and_reverse_dg_plot_validation():
    rng = np.random.default_rng(2023)
    ukln_shape = (47, 2, 2, 2000)
    dummy_ukln = rng.random(size=ukln_shape)

    with pytest.raises(AssertionError, match="fewer samples than frames_per_step"):
        plot_forward_and_reverse_dg(dummy_ukln, frames_per_step=ukln_shape[-1] + 1)
