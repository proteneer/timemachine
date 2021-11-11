import functools

from typing import Tuple
import jax
import numpy as np
import jax.numpy as jnp
from timemachine.potentials import bonded, rmsd
from scipy.stats import special_ortho_group


def exp_u(rotation, k, beta):
    return jnp.exp(-beta * rmsd.psi(rotation, k))


exp_batch = jax.jit(jax.vmap(exp_u, (0, None, None)))


def sample_multiple_rotations(k, beta, size):
    num_batches = 500
    batch_size = 10000
    state = np.random.RandomState(2021)
    samples = []
    for batch_attempt in range(num_batches):
        Rs = special_ortho_group.rvs(3, size=batch_size, random_state=state)
        tests = state.rand(batch_size)
        M = np.pi ** 2  # volume of SO(3)

        # (detailed explanation by jfass re: normalizing comments)
        # In rejection sampling, we need an upper bound M on the ratio q_target(x) / q_proposal(x),
        # so that q_target(x) / M q_proposal(x) <= 1 everywhere.
        # In our case we can get a tight bound from looking at q_target and q_proposal independently.
        #     q_target(x) = exp(-U(x)), and we know min_x U(x) = 0, so we know max_x q_target(x) = 1.
        #     Our proposal is uniform over rotations, so q_proposal(x) is a constant.

        # If we used normalized_q_proposal(x) = 1 / pi^2, then M=pi^2 would be an appropriate constant
        # to use, guaranteeing q_target(x) / M normalized_q_proposal(x) <= 1 everywhere.

        # We could just as well set q_proposal(x) = 1 and M=1, which has exactly the same effect.

        # However, mixing and matching these could lead to errors.

        # For example, it would be incorrect to use normalized_q_proposal(x) = 1 / pi^2, M=1, as we
        # could now sample x where q_target(x) / M normalized_q_proposal(x) > 1. Setting M too small
        # like this would impact the correctness of the sampler, but would also be easy to detect by
        # runtime assertion.

        # On the other hand, setting M larger than needed does not impact the correctness of the
        # sampler, although it impacts the efficiency. I think a previous version of this code set M ~10x
        # larger than it needed to be, by implicitly using q_proposal(x) = 1, and picking M = pi^2.
        # Setting M=1 here doesn't change the correctness of the sampler, but increases its acceptance
        # rate relative to that version.
        M = 1
        acceptance_prob = exp_batch(Rs, k, beta) / M
        locations = np.argwhere(tests < acceptance_prob).reshape(-1)

        samples.append(Rs[locations])
        if sum([len(x) for x in samples]) > size:
            break

    result = np.concatenate(samples)[:size]
    assert len(result) == size

    return result


def estimate_delta_us(k_translation, k_rotation, core_idxs, core_params, beta, lhs_xs, rhs_xs):
    """
    Compute the BAR re-weighted end-point correction of converting an intractable core
    restraint into a tractable RMSD-based orientational restraint.

    Parameters
    ----------
    k_translation: float
        Force constant of the translational restraint

    k_rotation: float
        Force constant of the rotational restraint

    core_idxs: int np.array (C, 2)
        Atom mapping between the two cores

    core_params: float np.array (C, 2)
        Bonded parameters of the intractable restraint

    lhs_xs: np.array [T, N, 3]
        Samples from the intractable left hand state, with the restraints turned on.

    rhs_xs: np.array [T, N, 3]
        Samples from the non-interacting, fully unrestrained right hand state.

    beta: 1/kT
        inverse kT

    Returns
    -------
    4-tuple
        4-tuple array of lhs delta_Us, rhs delta_Us, sampled rotations, and sampled translations.
        The sign of the delta_Us is defined to be consistent with that of pymbar. i.e. lhs_dU measures
        the difference U_rhs - U_lhs, using samples from lhs_xs. Whereas rhs_dU measures the difference
        U_lhs - U_rhs, using samples from rhs_xs.

    """
    # Setup a random state to ensure deterministic outputs
    state = np.random.RandomState(2021)
    box = np.eye(3) * 100.0
    core_restr = functools.partial(bonded.harmonic_bond, bond_idxs=core_idxs, params=core_params, box=box, lamb=None)

    # center of mass translational restraints
    restr_group_idxs_a = core_idxs[:, 0]
    restr_group_idxs_b = core_idxs[:, 1]

    # disjoint sets
    assert len(set(restr_group_idxs_a.tolist()).intersection(set(restr_group_idxs_b.tolist()))) == 0

    translation_restr = functools.partial(
        bonded.centroid_restraint,
        group_a_idxs=restr_group_idxs_a,
        group_b_idxs=restr_group_idxs_b,
        params=None,
        kb=k_translation,
        b0=0.0,
        box=box,
        lamb=None,
    )

    rotation_restr = functools.partial(
        rmsd.rmsd_restraint,
        params=None,
        group_a_idxs=restr_group_idxs_a,
        group_b_idxs=restr_group_idxs_b,
        k=k_rotation,
        box=box,
        lamb=None,
    )

    # (ytz): delta_U is simplified to this expression as the rest of the hamiltonian is unaffected
    # by a rigid translation/rotation. delta_U is expressed as rhs_U - lhs_U.
    def u_rhs(x_t):
        return translation_restr(x_t) + rotation_restr(x_t)

    def u_lhs(x_t):
        return core_restr(x_t)

    def delta_u_fwd_fn(x_t):
        return u_rhs(x_t) - u_lhs(x_t)

    def delta_u_rev_fn(x_t):
        return u_lhs(x_t) - u_rhs(x_t)

    delta_u_fwd_batch = jax.jit(jax.vmap(delta_u_fwd_fn))
    delta_u_rev_batch = jax.jit(jax.vmap(delta_u_rev_fn))

    lhs_du = delta_u_fwd_batch(lhs_xs)

    sample_size = rhs_xs.shape[0]
    rotation_samples = sample_multiple_rotations(k_rotation, beta, sample_size)
    covariance = np.eye(3) / (2 * beta * k_translation)
    translation_samples = state.multivariate_normal((0, 0, 0), covariance, sample_size)

    def align(x, r, t):
        x_a, x_b = rmsd.rmsd_align(x[restr_group_idxs_a], x[restr_group_idxs_b])
        x_b = x_b @ r.T + t
        x_new = jax.ops.index_update(x, restr_group_idxs_a, x_a)
        x_new = jax.ops.index_update(x_new, restr_group_idxs_b, x_b)
        return x_new

    batch_align_fn = jax.jit(jax.vmap(align, (0, 0, 0)))
    rhs_xs_aligned = batch_align_fn(rhs_xs, rotation_samples, translation_samples)
    rhs_du = delta_u_rev_batch(rhs_xs_aligned)

    return lhs_du, rhs_du, rotation_samples, translation_samples


# courtesy of jfass
def ecdf(x: np.array) -> Tuple[np.array, np.array]:
    """empirical cdf, from https://stackoverflow.com/a/37660583"""
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys


def overlap_from_cdf(w_f, w_r) -> float:
    """Quantify "overlap" between p_f(w) and p_r(-w)

    0: no overlap
    1: perfect overlap

    Notes
    -----
    * This function gives a more quantitative alternative to "eyeballing" the forward and reverse work histograms, but
        the returned quantity does not have an obvious statistical interpretation or tolerable threshold value.
    * See https://github.com/choderalab/pymbar/blob/3c4262c490261110a7595eec37df3e2b8caeab37/pymbar/bar.py#L344-L498
        for a discussion and implementation of an asymptotic variance estimate for BAR.
    * There are important cases where the work overlap computed from a single simulation run "looks good," but run-to-run
        variability is still large. (For example, if end-state sampling is intractable, and each run is trapped in a
        different metastable basin.) A work-overlap-based diagnostic like this will not be sensitive in these cases.
    """
    cdf_f = ecdf(w_f)
    cdf_r = ecdf(-w_r)

    # all work values
    combined = np.hstack([w_f, -w_r])

    # empirical cdf evaluated on all work values
    f = np.interp(combined, *cdf_f)
    r = np.interp(combined, *cdf_r)

    # max difference in cdfs :
    # * if the cdfs are identical this will be 0
    # * if the cdfs describe distributions with disjoint support, this will be 1
    max_cdf_difference = np.max(np.abs(f - r))

    return 1 - max_cdf_difference
