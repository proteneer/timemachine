import numpy as np
import pymbar
import pytest
from jax import grad, jit, value_and_grad, vmap
from jax import numpy as jnp

from timemachine.constants import BOLTZ
from timemachine.datasets import fetch_freesolv
from timemachine.fe.bar import DEFAULT_SOLVER_PROTOCOL, DG_KEY
from timemachine.fe.reweighting import (
    construct_endpoint_reweighting_estimator,
    construct_mixture_reweighting_estimator,
    interpret_as_mixture_potential,
    one_sided_exp,
)
from timemachine.ff import Forcefield
from timemachine.md.enhanced import get_solvent_phase_system
from timemachine.potentials import SummedPotential
from timemachine.testsystems.gaussian1d import make_gaussian_testsystem

pytestmark = [pytest.mark.nocuda]


def assert_estimator_accurate(estimate_delta_f, analytical_delta_f, ref_params, n_random_trials, atol):
    """for many random parameter sets, assert that the reweighted estimates of
    delta_f(params) and grad(delta_f)(params) are consistent with analytical_delta_f function

    Parameters
    ----------
    estimate_delta_f: Callable[[params], float]
        estimated free energy difference as a
        differentiable function of params
    analytical_delta_f: Callable[[params], float]
        exact free energy difference as a
        differentiable function of params
    ref_params: [2,] array
        generate random parameters near ref_params
    n_random_trials: int
    atol: float
        absolute tolerance
    """

    def sample_random_params():
        mean = ref_params[0] + np.random.rand()
        log_sigma = ref_params[1] - np.random.rand()
        return jnp.array([mean, log_sigma])

    f_hat, g_hat = value_and_grad(estimate_delta_f)(ref_params)
    f_ref, g_ref = value_and_grad(analytical_delta_f)(ref_params)

    np.testing.assert_allclose(f_hat, f_ref, atol=atol)
    np.testing.assert_allclose(g_hat, g_ref, atol=atol)

    for _ in range(n_random_trials):
        trial_params = sample_random_params()
        f_hat, g_hat = value_and_grad(estimate_delta_f)(trial_params)
        f_ref, g_ref = value_and_grad(analytical_delta_f)(trial_params)

        np.testing.assert_allclose(f_hat, f_ref, atol=atol)
        np.testing.assert_allclose(g_hat, g_ref, atol=atol)


def test_endpoint_reweighting_1d():
    """assert that endpoint reweighting estimator for delta_f(params), grad(delta_f)(params) is accurate
    on tractable 1D system"""
    np.random.seed(2022)

    u_fxn, _, sample, reduced_free_energy = make_gaussian_testsystem()

    # ref_params: (mean, log_sigma) @ lambda=1
    ref_params = np.ones(2)  # (annealing Normal(0, 1) @ lambda=0 to Normal(1, exp(1)) @ lambda=1)
    ref_delta_f = reduced_free_energy(1.0, ref_params) - reduced_free_energy(0.0, ref_params)

    # more samples --> tighter absolute tolerance possible in test assertion
    n_samples = int(1e6)
    atol = 1e-2

    samples_0 = sample(0, ref_params, n_samples)
    samples_1 = sample(1, ref_params, n_samples)

    vec_u = vmap(u_fxn, in_axes=(0, None, None))
    vec_u_0_fxn = lambda xs, params: vec_u(xs, 0, params)
    vec_u_1_fxn = lambda xs, params: vec_u(xs, 1, params)

    estimate_delta_f = construct_endpoint_reweighting_estimator(
        samples_0, samples_1, vec_u_0_fxn, vec_u_1_fxn, ref_params, ref_delta_f
    )
    analytical_delta_f = lambda params: reduced_free_energy(1.0, params) - reduced_free_energy(0.0, params)

    assert_estimator_accurate(jit(estimate_delta_f), analytical_delta_f, ref_params, n_random_trials=10, atol=atol)


def test_mixture_reweighting_1d():
    """using a variety of free energy estimates (MBAR, TI, analytical) to obtain reference mixture weights,
    assert that mixture reweighting estimator of delta_f(params), grad(delta_f)(params) is accurate
    on tractable 1D system"""
    np.random.seed(2022)

    u_fxn, normalized_u_fxn, sample, reduced_free_energy = make_gaussian_testsystem()

    # ref_params: (mean, log_sigma) @ lambda=1
    ref_params = np.ones(2)  # (annealing Normal(0, 1) @ lambda=0 to Normal(1, exp(1)) @ lambda=1)
    # easier-to-estimate free energy difference -> tighter tolerance possible in assertion

    n_windows = 10
    lambdas = np.linspace(0, 1, n_windows)

    # bigger n samples per window --> ~ sqrt(n)-tighter tolerance possible in assertion
    n_samples_per_window = int(1e5)
    atol = 1e-2

    N_k = np.array([n_samples_per_window] * n_windows)
    n_samples_total = sum(N_k)

    trajs = [sample(lam, ref_params, n_samples_per_window) for lam in lambdas]
    xs = jnp.vstack(trajs).flatten()
    u_kn = np.zeros((n_windows, n_samples_total))
    vec_u = vmap(u_fxn, in_axes=(0, None, None))

    # TODO [generality] : change signature to be (lam, params) instead of (xs, lam, params)?
    vec_u_0_fxn = lambda xs, params: vec_u(xs, 0.0, params)
    vec_u_1_fxn = lambda xs, params: vec_u(xs, 1.0, params)

    for k in range(n_windows):
        u_kn[k] = vec_u(xs, lambdas[k], ref_params)

    # f_k estimates can come from any source, as long as they're accurate

    # using analytical f_k at ref_params
    f_k_exact = jnp.array([reduced_free_energy(lam, ref_params) for lam in lambdas])
    u_mix_exact = interpret_as_mixture_potential(u_kn, f_k_exact, N_k)

    # various approximations to f_k at ref_params

    # MBAR
    mbar = pymbar.mbar.MBAR(u_kn, N_k=N_k, solver_protocol=DEFAULT_SOLVER_PROTOCOL)
    f_k_mbar = mbar.f_k
    u_mix_mbar = interpret_as_mixture_potential(u_kn, f_k_mbar, N_k)

    # TI
    vec_du_dl = vmap(grad(u_fxn, 1), (0, None, None))
    mean_du_dls = np.array([np.mean(vec_du_dl(traj, lam, ref_params)) for (traj, lam) in zip(trajs, lambdas)])
    f_k_ti = np.array([np.trapezoid(mean_du_dls[:k], lambdas[:k]) for k in range(n_windows)])
    u_mix_ti = interpret_as_mixture_potential(u_kn, f_k_ti, N_k)

    # TODO [overkill] : BAR
    # ...
    # u_mix_bar = ...

    # TODO [overkill] : SMC
    # ...
    # u_mix_smc = ...

    u_mixes = dict(exact_f_k=u_mix_exact, mbar_f_k=u_mix_mbar, ti_f_k=u_mix_ti)
    analytical_delta_f = lambda params: reduced_free_energy(1.0, params) - reduced_free_energy(0.0, params)

    for condition in u_mixes:
        u_mix = u_mixes[condition]

        estimate_delta_f = jit(construct_mixture_reweighting_estimator(xs, u_mix, vec_u_0_fxn, vec_u_1_fxn))

        assert_estimator_accurate(estimate_delta_f, analytical_delta_f, ref_params, n_random_trials=10, atol=atol)


def _make_fake_sample_batch(conf, box, ligand_indices, n_snapshots=25):
    """PURELY FOR TESTING -- get arrays that look like a batch of confs, boxes
    (but instead of actually populating confs, boxes with valid samples
     just randomly perturb conf and box a bunch of times)
    """

    samples = []

    for _ in range(n_snapshots):
        _conf = np.array(conf)
        _conf[ligand_indices] += 0.005 * np.random.randn(len(ligand_indices), 3)

        _box = box + np.diag(0.005 * np.random.randn(3))

        samples.append((_conf, _box))

    return samples


def make_ahfe_test_system():
    """an alchemical freesolv ligand in a water box, with:
    * batched, differentiable reduced potential functions
    * fake "endpoint samples" (random perturbations of initial (conf, box) -- not actual samples!)
    """
    mol = fetch_freesolv()[123]
    ff = Forcefield.load_default()
    temperature = 300
    ref_delta_f = -23.0  # from a short SMC calculation on mobley_242480, in kB T

    # doesn't have to be the same
    n_snapshots_0 = 10
    n_snapshots_1 = 20

    ubps, params, masses, conf, box = get_solvent_phase_system(mol, ff, 0.0)

    lambda_offset_idxs = ubps[-1].get_lambda_offset_idxs()
    ligand_indices = np.where(lambda_offset_idxs == 1)[0]
    ref_params = params[-1][ligand_indices]

    # pretend these are endpoint samples
    samples_0 = _make_fake_sample_batch(conf, box, ligand_indices, n_snapshots_0)
    samples_1 = _make_fake_sample_batch(conf, box, ligand_indices, n_snapshots_1)

    U_fxn = SummedPotential(ubps, params)

    def make_batched_u_fxn(lam=0.0):
        def batched_u_fxn(samples, ligand_nb_params):
            new_params = [jnp.array(p) for p in params]
            new_params[-1] = new_params[-1].at[ligand_indices].set(ligand_nb_params)

            U_s = jnp.array([U_fxn(conf, new_params, box, lam) for (conf, box) in samples])
            u_s = U_s / (BOLTZ * temperature)

            return u_s

        return batched_u_fxn

    batched_u_0 = make_batched_u_fxn(lam=0.0)
    batched_u_1 = make_batched_u_fxn(lam=1.0)

    return samples_0, samples_1, batched_u_0, batched_u_1, ref_params, ref_delta_f


@pytest.mark.skip(reason="needs update since removal of lambda dependence in nonbonded potentials")
def test_endpoint_reweighting_ahfe():
    """on made-up inputs of the right shape,
    check that derivative of an absolute hydration free energy w.r.t .ligand nonbonded parameters can be computed using
    custom ops
    """
    np.random.seed(2022)

    fake_samples_0, fake_samples_1, batched_u_0, batched_u_1, ref_params, ref_delta_f = make_ahfe_test_system()

    estimate_delta_f = construct_endpoint_reweighting_estimator(
        fake_samples_0, fake_samples_1, batched_u_0, batched_u_1, ref_params, ref_delta_f
    )

    v, g = value_and_grad(estimate_delta_f)(ref_params)

    assert np.isfinite(v)
    assert v == ref_delta_f
    assert np.isfinite(g).all()
    assert (g != 0).any()
    assert g.shape == ref_params.shape
    # assert anything_about_direction_of_g  # not expected because the "sample" arrays are made up

    # expect different estimate when evaluated on slightly different parameters
    params_prime = ref_params + 0.01 * np.random.randn(*ref_params.shape)
    v_prime, g_prime = value_and_grad(estimate_delta_f)(params_prime)
    assert v_prime != v
    assert (g_prime != g).any()
    assert np.isfinite(v_prime)
    assert np.isfinite(g_prime).all()


@pytest.mark.skip(reason="needs update since removal of lambda dependence in nonbonded potentials")
def test_mixture_reweighting_ahfe():
    """on made-up inputs of the right shape,
    check that derivative of an absolute hydration free energy w.r.t .ligand nonbonded parameters can be computed using
    custom ops
    """
    np.random.seed(2022)

    _samples_0, _samples_1, batched_u_0, batched_u_1, ref_params, ref_delta_f = make_ahfe_test_system()
    fake_samples_n = _samples_1 + _samples_1
    fake_u_ref_n = np.random.randn(len(fake_samples_n))

    estimate_delta_f = construct_mixture_reweighting_estimator(fake_samples_n, fake_u_ref_n, batched_u_0, batched_u_1)

    v, g = value_and_grad(estimate_delta_f)(ref_params)

    assert np.isfinite(v)
    # assert v == ref_delta_f  # not expected in this case, due to non-physical fake_log_weights
    assert np.isfinite(g).all()
    assert (g != 0).any()
    assert g.shape == ref_params.shape
    # assert anything_about_direction_of_g  # not expected because the inputs are non-physical

    # expect different estimate when evaluated on slightly different parameters
    params_prime = ref_params + 0.01 * np.random.randn(*ref_params.shape)
    v_prime, g_prime = value_and_grad(estimate_delta_f)(params_prime)
    assert v_prime != v
    assert (g_prime != g).any()
    assert np.isfinite(v_prime)
    assert np.isfinite(g_prime).all()


def test_one_sided_exp():
    """assert consistency with pymbar.exp on random instances + instances containing +inf work"""

    np.random.seed(2022)
    num_instances = 100

    for _ in range(num_instances):
        # instance parameters
        num_works = np.random.randint(1, 100)
        mean = np.random.randn() * 10
        stddev = np.exp(np.random.randn())

        # random instance
        reduced_works = np.random.randn(num_works) * stddev + mean

        # compare estimates
        pymbar_estimate = pymbar.exp(reduced_works)[DG_KEY]
        tm_estimate = one_sided_exp(reduced_works)

        assert np.isclose(tm_estimate, pymbar_estimate)

    # also check +inf
    reduced_works = jnp.array([+np.inf, 0])
    assert np.isclose(one_sided_exp(reduced_works), pymbar.exp(reduced_works)[DG_KEY])


def test_interpret_as_mixture_potential():
    """assert approximate self-consistency a la https://arxiv.org/abs/1704.00891

    Notes
    -----
    * interprets samples from multiple lambda windows as instead coming from a single mixture distribution
    * asserts that one-sided exp estimates of delta_f(mixture distribution -> lambda_window) are approximately
        equal to their exact values
    * this form of self-consistency should become exact in limit of large num_samples
    """

    np.random.seed(2022)

    # more samples --> tighter absolute tolerance possible in test assertion
    n_samples_per_window = int(1e6)
    atol = 1e-3

    u_fxn, normalized_u_fxn, sample, reduced_free_energy = make_gaussian_testsystem()

    ref_params = np.ones(2)
    n_windows = 5
    lambdas = np.linspace(0, 1, n_windows)

    N_k = [n_samples_per_window] * n_windows
    n_samples_total = sum(N_k)

    def make_arrays(normalized=False):
        """u_kn, f_k, N_k (with f_k = zeros if normalized)"""
        trajs = [sample(lam, ref_params, n_samples_per_window) for lam in lambdas]
        xs = jnp.vstack(trajs).flatten()

        u_kn = np.zeros((n_windows, n_samples_total))

        if normalized:
            vec_u = vmap(normalized_u_fxn, in_axes=(0, None, None))
            f_k = np.zeros(n_windows)
        else:
            vec_u = vmap(u_fxn, in_axes=(0, None, None))
            f_k = jnp.array([reduced_free_energy(lam, ref_params) for lam in lambdas])
            f_k -= f_k[0]

            # double-check this is different from normalized case
            assert (jnp.abs(f_k) > 10 * atol).any()

        for k in range(n_windows):
            u_kn[k] = vec_u(xs, lambdas[k], ref_params)

        return u_kn, f_k, N_k

    def reweight_from_mixture(u_kn, f_k, N_k):
        """https://arxiv.org/abs/1704.00891"""
        mixture_u_n = interpret_as_mixture_potential(u_kn, f_k, N_k)
        delta_u_kn = u_kn - mixture_u_n[jnp.newaxis, :]
        estimated_f_k = vmap(one_sided_exp)(delta_u_kn)
        return estimated_f_k - estimated_f_k[0]

    for normalized in [False, True]:
        u_kn, f_k, N_k = make_arrays(normalized)

        # double-check shape
        mixture_u_n = interpret_as_mixture_potential(u_kn, f_k, N_k)
        assert mixture_u_n.shape == (sum(N_k),)

        # if we reweight from mixture, should approximately recover component free energies
        estimated_f_k = reweight_from_mixture(u_kn, f_k, N_k)
        np.testing.assert_allclose(estimated_f_k, f_k, atol=atol)
