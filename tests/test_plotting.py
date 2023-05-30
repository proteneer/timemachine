import numpy as np
import pytest

from timemachine.fe.plots import plot_forward_and_reverse_ddg


@pytest.mark.nogpu
def test_forward_and_reverse_ddg_plot():
    rng = np.random.default_rng(2023)
    ukln_shape = (47, 2, 2, 2000)
    dummy_solv_ukln = rng.random(size=ukln_shape)
    dummy_complex_ukln = rng.random(size=ukln_shape)

    plot_forward_and_reverse_ddg(dummy_solv_ukln, dummy_complex_ukln)
