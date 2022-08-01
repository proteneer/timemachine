import numpy as np
from jax import numpy as jnp

from timemachine.potentials.jax_utils import bernoulli_logpdf


def local_resampling_move(
    x,
    target_logpdf_fxn,
    particle_selection_log_prob_fxn,
    mcmc_move,
):
    x = jnp.array(x)
    n_particles = len(x)

    # select particles to be updated
    selection_probs = np.exp(particle_selection_log_prob_fxn(x))
    assert np.min(selection_probs) >= 0 and np.max(selection_probs) <= 1, "selection_probs must be in [0,1]"
    assert selection_probs.shape == (n_particles,), "must compute per-particle selection_probs"
    selection_mask = np.random.rand(n_particles) < selection_probs  # TODO: factor out dependence on global numpy rng?

    # construct restrained version of target
    def restrained_logpdf_fxn(x) -> float:
        log_p_i = particle_selection_log_prob_fxn(x)
        return target_logpdf_fxn(x) + bernoulli_logpdf(log_p_i, selection_mask)

    # construct smaller sampling problem, defined only on selected particles
    def subproblem_logpdf(x_sub) -> float:
        x_full = x.at[selection_mask].set(x_sub)
        return restrained_logpdf_fxn(x_full)

    # apply any valid MCMC move to this subproblem
    x_sub = x[selection_mask]
    x_next_sub, aux = mcmc_move(x_sub, subproblem_logpdf)
    x_next = x.at[selection_mask].set(x_next_sub)

    return x_next, aux
