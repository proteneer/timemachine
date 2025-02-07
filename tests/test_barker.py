"""
Assert that the move:

    Is correctly implemented:
    - [x] Methods sample(x) and log_density(x, y) support expected shapes
    - [x] Proposal is normalized
    - [x] An MCMC move constructed from this proposal accurately samples a simple target

    Has robust behavior:
    - [x] Resolves clashes without blowing up or requiring alchemical intermediates
        (Added to test_minimizer.py)
    - [x] Avg. norm(proposal - x) remains ~= avg. norm(gaussian(0, sig)), even when norm(grad_log_q(x)) >> 1
"""

import numpy as np
import pytest
from jax import grad, jit
from jax import numpy as jnp

from timemachine.md.barker import BarkerProposal

pytestmark = [pytest.mark.nocuda]


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
        return -4 * x**3

    y_grid = np.linspace(-10, +10, 1_000)

    prop = BarkerProposal(grad_log_q, proposal_sig=proposal_sig)
    logpdf_grid = np.array([prop.log_density(x0, y) for y in y_grid])
    pdf_grid = np.exp(logpdf_grid)

    Z = np.trapezoid(pdf_grid, y_grid)

    assert Z == pytest.approx(1)


def test_accurate_mcmc(threshold=1e-4):
    def log_q(x):
        return np.sum(-(x**4))

    def grad_log_q(x):
        return -4 * x**3

    rng = np.random.default_rng(0)

    # system with a large number of quartic oscillators
    x = np.zeros(1_000)

    prop = BarkerProposal(grad_log_q, proposal_sig=0.1, seed=0)

    def mcmc_move(x):
        y = prop.sample(x)

        log_prob_fwd = prop.log_density(x, y)
        log_prob_rev = prop.log_density(y, x)

        _log_accept_prob = log_q(y) - log_q(x) + log_prob_rev - log_prob_fwd
        accept_prob = np.exp(min(0.0, _log_accept_prob))

        if rng.random() < accept_prob:
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
    y_ref = y / np.trapezoid(y, x_grid)

    histogram_mse = np.mean((y_ref - y_empirical) ** 2)

    assert histogram_mse < threshold


@pytest.mark.parametrize("proposal_sig", [0.1, 1.0])
@pytest.mark.parametrize("seed", range(5))
def test_proposal_magnitude_independent_of_force_magnitude(proposal_sig, seed):
    """Generate Lennard-Jones-informed proposals from clashy vs. relaxed starting points
        (where |force(x_clash)| ~= +inf and |force(x_relaxed)| ~ 0).

    Assert that:
    * the avg. squared distance between proposal and starting point is the same in both cases
    * the proposal skew is ~ 100% in the |force| ~= +inf case, and ~ 0% in the |force| ~= 0 case
    """

    def log_q(r):
        sig, eps = 1.0, 1.0
        return jnp.sum(-4 * eps * ((sig / r) ** 12 - (sig / r) ** 6))

    grad_log_q = jit(grad(log_q))

    expected_sq_distance = proposal_sig**2
    n_samples = 100_000
    rel_tol = 1e-2
    abs_tol = 1e-2

    # ---------------------
    barker_prop = BarkerProposal(grad_log_q, proposal_sig=proposal_sig, seed=seed)

    # sample many proposals from a clashy initial condition
    x_clash = 1e-3 * np.ones(n_samples)
    assert np.linalg.norm(grad_log_q(x_clash)) > 1e10
    ys_clash = barker_prop.sample(x_clash)

    # assert that the gradient-informed proposals
    # are the same avg. sq. distance from starting point
    # as if proposed from the base kernel Normal(mu=x_clash, sig=proposal_sig)
    disp_clash = (ys_clash - x_clash).flatten()
    mean_sq_distance_clash = (disp_clash**2).mean()

    assert mean_sq_distance_clash == pytest.approx(expected_sq_distance, rel=rel_tol)

    # assert that the proposals are skewed in the expected direction
    skew = np.sign(disp_clash).mean()
    assert skew == pytest.approx(1, abs=abs_tol)

    # ---------------------
    barker_prop = BarkerProposal(grad_log_q, proposal_sig=proposal_sig, seed=seed)

    # sample many proposals from a relaxed initial condition
    x_relaxed = 1e3 * np.ones(n_samples)
    assert np.linalg.norm(grad_log_q(x_relaxed)) < 1e-10
    ys = barker_prop.sample(x_relaxed)

    # again, assert proposals are the expected avg. sq. distance from starting point
    disp = (ys - x_relaxed).flatten()
    mean_sq_distance = (disp**2).mean()
    assert mean_sq_distance == pytest.approx(expected_sq_distance, rel=rel_tol)

    # assert that the proposals are not skewed much
    skew = np.sign(disp).mean()
    assert skew == pytest.approx(0, abs=abs_tol)

    # ---------------------

    # shared random seed, so can assert with tighter tolerance that (A == B)
    # than we could assert ((A == expectation) and (B == expectation)) above
    assert pytest.approx(mean_sq_distance, rel=1e-100) == mean_sq_distance_clash
