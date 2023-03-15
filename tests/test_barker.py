"""
Ideas for tests:

Assert that the move:

    Is correctly implemented:
    - [x] Methods sample(x) and log_density(x, y) support expected shapes
    - [x] Proposal is normalized
    - [ ] An MCMC move constructed from this proposal accurately samples a simple target

    Has robust behavior:
    - [x] Resolves clashes without blowing up or requiring alchemical intermediates
        (Added to test_minimizer.py)
    - [ ] Avg. norm(proposal - x) remains <= avg. norm(gaussian(0, sig)), even when norm(grad_log_q(x)) >> 1
"""

import numpy as np
import pytest

from timemachine.md.barker import BarkerProposal


def test_barker_shapes():
    def grad_log_q(x):
        return np.ones_like(x)

    prop = BarkerProposal(grad_log_q)

    shapes = [(1,), (10,), (10, 3)]

    for shape in shapes:
        x = np.ones(shape)
        y = prop.sample(x)

        assert x.shape == y.shape == shape
        logpdf = prop.log_density(x, y)

        assert np.isscalar(logpdf)


@pytest.mark.parametrize("x0", [-1, 0, +1])
@pytest.mark.parametrize("proposal_sig", [0.1, 1.0])
def test_proposal_normalization(x0, proposal_sig):
    r"""numerically integrate \int dy p_sig(y | x0) and assert close to 1"""

    def grad_log_q(x):
        return -4 * x ** 3

    y_grid = np.linspace(-10, +10, 1_000)

    prop = BarkerProposal(grad_log_q, proposal_sig=proposal_sig)
    logpdf_grid = np.array([prop.log_density(x0, y) for y in y_grid])
    pdf_grid = np.exp(logpdf_grid)

    Z = np.trapz(pdf_grid, y_grid)

    assert pytest.approx(Z) == 1
