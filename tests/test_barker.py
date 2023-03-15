"""
Ideas for tests:

Assert that the move:

    Is correctly implemented:
    - [x] Methods sample(x) and log_density(x, y) support expected shapes
    - [x] Proposal is normalized
    - [x] An MCMC move constructed from this proposal accurately samples a simple target

    Has robust behavior:
    - [x] Resolves clashes without blowing up or requiring alchemical intermediates
        (Added to test_minimizer.py)
    - [ ] Avg. norm(proposal - x) remains <= avg. norm(gaussian(0, sig)), even when norm(grad_log_q(x)) >> 1
"""

import numpy as np
import pytest

from timemachine.md.barker import BarkerProposal


@pytest.mark.nogpu
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


@pytest.mark.nogpu
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


@pytest.mark.nogpu
def test_accurate_mcmc(threshold=1e-4):
    np.random.seed(0)

    def log_q(x):
        return np.sum(-(x ** 4))

    def grad_log_q(x):
        return -4 * x ** 3

    # system with a large number of quartic oscillators
    x = np.zeros(1_000)

    prop = BarkerProposal(grad_log_q, proposal_sig=0.1)

    def mcmc_move(x):
        y = prop.sample(x)

        log_prob_fwd = prop.log_density(x, y)
        log_prob_rev = prop.log_density(y, x)

        _log_accept_prob = log_q(y) - log_q(x) + log_prob_rev - log_prob_fwd
        accept_prob = np.exp(min(0.0, _log_accept_prob))

        if np.random.rand() < accept_prob:
            return y
        else:
            return x

    _traj = [x]
    for _ in range(2_000):
        _traj.append(mcmc_move(_traj[-1]))

    samples = np.array(_traj[100:]).flatten()

    # summarize using histogram
    y_empirical, edges = np.histogram(samples, bins=100, range=(-2, +2), density=True)
    x_grid = (edges[1:] + edges[:-1]) / 2

    # compare with ref
    y = np.exp(np.array([log_q(x) for x in x_grid]))
    y_ref = y / np.trapz(y, x_grid)

    histogram_mse = np.mean((y_ref - y_empirical) ** 2)

    assert histogram_mse < threshold
