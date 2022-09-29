import numpy as np
import pymbar
import pytest
from pymbar.testsystems import gaussian_work_example

from timemachine.fe.bar import bar_with_bootstrapped_uncertainty, bootstrap_bar


@pytest.mark.nogpu
def test_bootstrap_bar():
    np.random.seed(0)
    n_bootstrap = 1000

    for sigma_F in [0.1, 1, 10]:
        # default rbfe instance size, varying difficulty
        w_F, w_R = gaussian_work_example(2000, 2000, sigma_F=sigma_F, seed=0)

        # estimate 3 times
        df_ref, ddf_ref = pymbar.BAR(w_F, w_R)
        df_0, bootstrap_samples = bootstrap_bar(w_F, w_R, n_bootstrap=n_bootstrap)
        df_1, bootstrap_sigma = bar_with_bootstrapped_uncertainty(w_F, w_R)

        # assert estimates identical, uncertainties comparable
        print(f"stddev(w_F) = {sigma_F}, bootstrap uncertainty = {bootstrap_sigma}, pymbar.BAR uncertainty = {ddf_ref}")
        assert df_0 == df_ref
        assert df_1 == df_ref
        assert len(bootstrap_samples) == n_bootstrap, "timed out on default problem size!"
        np.testing.assert_approx_equal(bootstrap_sigma, ddf_ref, significant=1)
