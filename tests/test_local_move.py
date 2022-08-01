from jax import config, grad, jit
from jax import numpy as jnp
from jax import vmap

config.update("jax_enable_x64", True)

from functools import partial

import numpy as np
import pytest

from timemachine.integrator import VelocityVerletIntegrator
from timemachine.md.local_resampling import local_resampling_move
from timemachine.potentials.jax_utils import delta_r


def make_hmc_mover(x, logpdf_fxn, dt=0.1, n_steps=100):
    masses = np.ones(len(x))

    def force_fxn(x):
        return -grad(logpdf_fxn)(x)

    integrator = VelocityVerletIntegrator(force_fxn, masses=masses, dt=dt)

    def augmented_logpdf(x, v):
        return logpdf_fxn(x) - np.sum(0.5 * masses[:, np.newaxis] * v ** 2)

    @jit
    def augmented_proposal(x0, v0):
        logp_before = augmented_logpdf(x0, v0)
        x1, v1 = integrator._update_via_fori_loop(x0, v0, n_steps=n_steps)
        logp_after = augmented_logpdf(x1, v1)

        log_accept_prob = jnp.clip(jnp.nan_to_num(logp_after - logp_before, nan=-np.inf), a_max=0.0)

        return (x1, v1), log_accept_prob

    def hmc_move(x0):
        v0 = np.random.randn(*x.shape)
        (x1, v1), log_accept_prob = augmented_proposal(x0, v0)

        if np.random.rand() < jnp.exp(log_accept_prob):
            return x1, log_accept_prob
        else:
            return x0, log_accept_prob

    return hmc_move


def expect_no_drift(x0, move_fxn, observable_fxn, n_local_resampling_iterations=100):
    traj = [jnp.array(x0)]
    aux_traj = []

    for _ in range(n_local_resampling_iterations):
        updated, aux = move_fxn(traj[-1])

        traj.append(updated)
        aux_traj.append(aux)

    expected_selection_fraction_traj = np.array([observable_fxn(x) for x in traj])

    # TODO: don't hard-code T, thresholds, etc.
    T = 10
    assert n_local_resampling_iterations > 2 * T
    avg_at_start = np.mean(expected_selection_fraction_traj[:T])
    avg_at_end = np.mean(expected_selection_fraction_traj[-T:])

    deviated_by_50percent_or_more = (avg_at_start / avg_at_end) <= 0.5 or (avg_at_start / avg_at_end) >= 1.5
    if deviated_by_50percent_or_more:
        msg = f"""
            observable avg over start frames = {avg_at_start:.3f}
            observable avg over end frames = {avg_at_end:.3f}
            but averages of this (and all other observables) should be constant over time
        """
        raise RuntimeError(msg)

    return traj, aux_traj


def naive_local_resampling_move(
    x,
    target_logpdf_fxn,
    particle_selection_log_prob_fxn,
    mcmc_move,
):
    """WARNING: deliberately incorrect, with a key step ablated for testing purposes!

    local_resampling_move, but with restraint potential disabled"""
    n_particles = len(x)

    # select particles to be updated
    selection_probs = np.exp(particle_selection_log_prob_fxn(x))
    assert np.min(selection_probs) >= 0 and np.max(selection_probs) <= 1, "selection_probs must be in [0,1]"
    assert selection_probs.shape == (n_particles,), "must compute per-particle selection_probs"
    selection_mask = np.random.rand(n_particles) < selection_probs  # TODO: factor out dependence on global numpy rng?

    # NOTE: missing restraint! will result in incorrect sampling

    # def restrained_logpdf_fxn(x) -> float:
    #    log_p_i = particle_selection_log_prob_fxn(x)
    #    return target_logpdf_fxn(x) + bernoulli_logpdf(log_p_i, selection_mask)

    def subproblem_logpdf(x_sub) -> float:
        x_full = x.at[selection_mask].set(x_sub)
        return target_logpdf_fxn(x_full)  # return restrained_logpdf_fxn(x_full)

    # apply any valid MCMC move to this subproblem
    x_sub = x[selection_mask]
    x_next_sub, aux = mcmc_move(x_sub, subproblem_logpdf)
    x_next = x.at[selection_mask].set(x_next_sub)

    return x_next, aux


def test_ideal_gas():
    """Run HMC on subsets of an ideal gas system, where the subsets are selected based on a geometric criterion.
    Assert that an observable based on this criterion doesn't drift after a large number of updates when using
    local_resampling_move, and assert that it does drift when using an ablated version of local_resampling_move."""
    np.random.seed(2022)

    # make 2D ideal gas system
    box_size = 5
    dim = 2

    box = np.eye(dim) * box_size
    n_particles = 1000

    def ideal_gas_2d_logpdf_fxn(x):
        return 0.0

    x0 = np.random.rand(n_particles, 2) * box_size

    # make function that preferentially selects particles near center
    center = np.ones(dim) * (box_size / 2)
    r0 = box_size / 6

    def central_particle_selection_log_prob_fxn(x):
        distance_from_center = lambda x_i: jnp.linalg.norm(delta_r(x_i, center, box))
        r = vmap(distance_from_center)(x)

        return jnp.where(r > r0, -10 * (r - r0) ** 4, 0.0)

    # define any correct MCMC move fxn -- e.g. a composition of other correct MCMC moves
    def run_multiple_hmc_moves(x, logpdf_fxn, dt=1e-4, n_steps=100, n_moves=100):
        hmc_move = make_hmc_mover(x, logpdf_fxn, dt=dt, n_steps=n_steps)
        log_accept_probs = []
        for _ in range(n_moves):
            x, log_accept_prob = hmc_move(x)
            log_accept_probs.append(log_accept_prob)

        return x, np.array(log_accept_probs)

    # define correct and incorrect version of local move
    # (with the same target, same particle selection method, and same MCMC move)
    common_kwargs = dict(
        target_logpdf_fxn=ideal_gas_2d_logpdf_fxn,
        particle_selection_log_prob_fxn=central_particle_selection_log_prob_fxn,
        mcmc_move=run_multiple_hmc_moves,
    )
    correct_local_move = partial(local_resampling_move, **common_kwargs)
    incorrect_local_move = partial(naive_local_resampling_move, **common_kwargs)

    # test that num particles near center doesn't change dramatically
    def num_particles_near_center(x):
        return np.mean(np.exp(central_particle_selection_log_prob_fxn(x)))

    def assert_correctness(local_move):
        # expect no drift
        traj, aux_traj = expect_no_drift(
            x0, local_move, observable_fxn=num_particles_near_center, n_local_resampling_iterations=100
        )

        # assert move was not trivial
        avg_accept_prob = np.mean([np.mean(np.exp(log_accept_probs)) for log_accept_probs in aux_traj])
        assert avg_accept_prob > 0.1
        assert np.max(np.abs(traj[0] - traj[-1])) > r0

    # expect local move to be correct
    assert_correctness(correct_local_move)

    # expect failure with ablated version of local move
    with pytest.raises(RuntimeError):
        assert_correctness(incorrect_local_move)
