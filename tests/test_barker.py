"""
Ideas for tests:

Assert that the move:

    Is correctly implemented:
    - [ ] Methods sample(x), batch_sample(xs), log_density(x, y), batch_log_density(xs, ys) support expected shapes
    - [x] Proposal is normalized
    - [ ] An MCMC move constructed from this proposal accurately samples a simple target

    Has robust behavior:
    - [ ] Resolves clashes without blowing up or requiring alchemical intermediates
    - [ ] Avg. norm(proposal - x) remains <= avg. norm(gaussian(0, sig)), even when norm(grad_log_q(x)) >> 1
"""


import numpy as np
import pytest
from jax import vmap

from timemachine.md.barker import BarkerProposal


@pytest.mark.parameterize("x0", [-2, -1, 0, +1, +2])
@pytest.mark.parametrize("proposal_sig", [0.1, 0.5, 1.0])
def test_proposal_normalization(x0, proposal_sig):
    r"""numerically integrate \int dy p_sig(y | x0) and assert close to 1"""

    def grad_log_q(x):
        return -4 * x ** 3

    y_grid = np.linspace(-10, +10, 10_000)

    prop = BarkerProposal(grad_log_q, proposal_sig=proposal_sig)

    def f(y):
        return prop.log_density(x0, y)

    logpdf_grid = vmap(f)(y_grid)
    pdf_grid = np.exp(logpdf_grid)

    Z = np.trapz(pdf_grid, y_grid)

    assert pytest.approx(Z) == 1
