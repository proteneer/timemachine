import numpy as np

from timemachine.constants import DEFAULT_TEMP
from timemachine.fe.plots import plot_forward_and_reverse_ddg


def test_forward_and_reverse_ddg_plot():
    rng = np.random.default_rng(2023)
    ukln_shape = (47, 2, 2, 2000)
    dummy_solv_ukln = rng.random(size=ukln_shape)
    dummy_complex_ukln = rng.random(size=ukln_shape)

    plot_forward_and_reverse_ddg(dummy_solv_ukln, dummy_complex_ukln, DEFAULT_TEMP)
