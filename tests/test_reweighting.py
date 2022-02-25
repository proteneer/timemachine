import numpy as onp
import pymbar
from jax import grad, jit
from jax import numpy as np
from jax import value_and_grad, vmap
from jax.scipy.stats.norm import logpdf as norm_logpdf
from scipy.stats import norm

from timemachine.constants import BOLTZ
from timemachine.datasets import fetch_freesolv
from timemachine.fe.functional import construct_differentiable_interface_fast
from timemachine.fe.reweighting import (
    construct_endpoint_reweighting_estimator,
    construct_mixture_reweighting_estimator,
    interpret_as_mixture_potential,
    one_sided_exp,
)
from timemachine.ff import Forcefield
from timemachine.md.enhanced import get_solvent_phase_system

# TODO: should these be moved into a test fixture? or to the testsystems module?


def make_gaussian_testsystem():
    def annealed_gaussian_def(lam, params):
        initial_mean, initial_log_sigma = 0.0, 0.0
        target_mean, target_log_sigma = params

        # lam = 0 -> (mean = 0, stddev = 1)
        # lam = 1 -> (mean = target_mean, stddev = target_sigma)
        mean = lam * target_mean - (1 - lam) * initial_mean
        stddev = np.exp(lam * target_log_sigma + (1 - lam) * initial_log_sigma)

        return mean, stddev

    def sample(lam, params, n_samples):
        mean, stddev = annealed_gaussian_def(lam, params)
        return norm.rvs(loc=mean, scale=stddev, size=(n_samples, 1))

    def logpdf(x, lam, params):
        mean, stddev = annealed_gaussian_def(lam, params)
        return np.sum(norm_logpdf(x, loc=mean, scale=stddev))

    def u_fxn(x, lam, params):
        """unnormalized version of -logpdf"""
        mean, stddev = annealed_gaussian_def(lam, params)
        return np.sum(0.5 * ((x - mean) / stddev) ** 2)

    def normalized_u_fxn(x, lam, params):
        return -logpdf(x, lam, params)

    def reduced_free_energy(lam, params):
        mean, stddev = annealed_gaussian_def(lam, params)
        log_z = np.log(stddev * np.sqrt(2 * np.pi))
        return -log_z

    return u_fxn, normalized_u_fxn, sample, reduced_free_energy


def assert_estimator_accurate(estimate_delta_f, analytical_delta_f, ref_params, n_random_trials, atol=5e-3):
    """for many random parameter sets, assert that the reweighted estimates of
    delta_f(params) and grad(delta_f)(params) are consistent with analytical result"""

    def sample_random_params():
        mean = ref_params[0] + onp.random.rand()
        log_sigma = ref_params[1] - onp.random.rand()
        return np.array([mean, log_sigma])

    f_hat, g_hat = value_and_grad(estimate_delta_f)(ref_params)
    f_ref, g_ref = value_and_grad(analytical_delta_f)(ref_params)

    onp.testing.assert_allclose(f_hat, f_ref, atol=atol)
    onp.testing.assert_allclose(g_hat, g_ref, atol=atol)

    for _ in range(n_random_trials):
        trial_params = sample_random_params()
        f_hat, g_hat = value_and_grad(estimate_delta_f)(trial_params)
        f_ref, g_ref = value_and_grad(analytical_delta_f)(trial_params)

        onp.testing.assert_allclose(f_hat, f_ref, atol=atol)
        onp.testing.assert_allclose(g_hat, g_ref, atol=atol)


def test_endpoint_reweighting_1d():

    u_fxn, _, sample, reduced_free_energy = make_gaussian_testsystem()

    onp.random.seed(2022)

    ref_params = np.zeros(2)
    ref_delta_f = reduced_free_energy(1.0, ref_params) - reduced_free_energy(0.0, ref_params)

    n_samples = int(1e6)

    samples_0 = sample(0, ref_params, n_samples)
    samples_1 = sample(1, ref_params, n_samples)

    vec_u = vmap(u_fxn, in_axes=(0, None, None))
    vec_u_0_fxn = lambda xs, params: vec_u(xs, 0, params)
    vec_u_1_fxn = lambda xs, params: vec_u(xs, 1, params)

    estimate_delta_f = construct_endpoint_reweighting_estimator(
        samples_0, samples_1, vec_u_0_fxn, vec_u_1_fxn, ref_params, ref_delta_f
    )
    analytical_delta_f = lambda params: reduced_free_energy(1.0, params) - reduced_free_energy(0.0, params)

    assert_estimator_accurate(jit(estimate_delta_f), analytical_delta_f, ref_params, n_random_trials=10, atol=5e-3)


def _make_fake_sample_batch(conf, box, ligand_indices, n_snapshots=100):
    """PURELY FOR TESTING -- get arrays that look like a batch of confs, boxes
    (but instead of actually populating confs, boxes with valid samples
     just randomly perturb conf and box a bunch of times)
    """

    samples = []

    for _ in range(n_snapshots):
        _conf = onp.array(conf)
        _conf[ligand_indices] += 0.005 * onp.random.randn(len(ligand_indices), 3)

        _box = box + np.diag(0.005 * onp.random.randn(3))

        samples.append((_conf, _box))

    return samples


def make_ahfe_test_system():
    """an alchemical freesolv ligand in a water box, with:
    * batched, differentiable reduced potential functions (using construct_differentiable_interface_fast)
    * fake "endpoint samples" (random perturbations of initial (conf, box) -- not actual samples!)
    """
    mol = fetch_freesolv()[123]
    ff = Forcefield.load_from_file("smirnoff_1_1_0_ccc.py")
    temperature = 300
    ref_delta_f = -23.0  # from a short SMC calculation on mobley_242480, in kB T

    # doesn't have to be the same
    n_snapshots_0 = 100
    n_snapshots_1 = 50

    ubps, params, masses, conf, box = get_solvent_phase_system(mol, ff)

    lambda_offset_idxs = ubps[-1].get_lambda_offset_idxs()
    ligand_indices = onp.where(lambda_offset_idxs == 1)[0]
    ref_params = params[-1][ligand_indices]

    # pretend these are endpoint samples
    samples_0 = _make_fake_sample_batch(conf, box, ligand_indices, n_snapshots_0)
    samples_1 = _make_fake_sample_batch(conf, box, ligand_indices, n_snapshots_1)

    U_fxn = construct_differentiable_interface_fast(unbound_potentials=ubps, params=params)

    def make_batched_u_fxn(lam=0.0):
        def batched_u_fxn(samples, ligand_nb_params):
            new_params = [np.array(p) for p in params]
            new_params[-1] = new_params[-1].at[ligand_indices].set(ligand_nb_params)

            U_s = np.array([U_fxn(conf, new_params, box, lam) for (conf, box) in samples])
            u_s = U_s / (BOLTZ * temperature)

            return u_s

        return batched_u_fxn

    batched_u_0 = make_batched_u_fxn(lam=0.0)
    batched_u_1 = make_batched_u_fxn(lam=1.0)

    return samples_0, samples_1, batched_u_0, batched_u_1, ref_params, ref_delta_f


def test_mixture_reweighting_1d():
    """using a variety of free energy estimates (MBAR, TI, analytical) to obtain reference mixture weights,
    assert that mixture reweighting estimator of delta_f(params), grad(delta_f)(params) is accurate"""
    onp.random.seed(2022)

    u_fxn, normalized_u_fxn, sample, reduced_free_energy = make_gaussian_testsystem()

    ref_params = np.ones(2)
    n_windows = 10
    lambdas = np.linspace(0, 1, n_windows)

    n_samples_per_window = int(1e5)
    N_k = [n_samples_per_window] * n_windows
    n_samples_total = sum(N_k)

    trajs = [sample(lam, ref_params, n_samples_per_window) for lam in lambdas]
    xs = np.vstack(trajs).flatten()
    u_kn = onp.zeros((n_windows, n_samples_total))
    vec_u = vmap(u_fxn, in_axes=(0, None, None))

    # TODO [generality] : change signature to be (lam, params) instead of (xs, lam, params)?
    vec_u_0_fxn = lambda xs, params: vec_u(xs, 0.0, params)
    vec_u_1_fxn = lambda xs, params: vec_u(xs, 1.0, params)

    for k in range(n_windows):
        u_kn[k] = vec_u(xs, lambdas[k], ref_params)

    # f_k estimates can come from any source, as long as they're accurate

    # using analytical f_k at ref_params
    f_k_exact = np.array([reduced_free_energy(lam, ref_params) for lam in lambdas])
    u_mix_exact = interpret_as_mixture_potential(u_kn, f_k_exact, N_k)

    # various approximations to f_k at ref_params

    # MBAR
    mbar = pymbar.MBAR(u_kn, N_k=N_k)
    f_k_mbar = mbar.f_k
    u_mix_mbar = interpret_as_mixture_potential(u_kn, f_k_mbar, N_k)

    # TI
    vec_du_dl = vmap(grad(u_fxn, 1), (0, None, None))
    mean_du_dls = np.array([np.mean(vec_du_dl(traj, lam, ref_params)) for (traj, lam) in zip(trajs, lambdas)])
    f_k_ti = np.array([np.trapz(mean_du_dls[:k], lambdas[:k]) for k in range(n_windows)])
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

        # TODO [sign convention]: change signature to accept u_mix rather than log_weights?
        estimate_delta_f = jit(construct_mixture_reweighting_estimator(xs, -u_mix, vec_u_0_fxn, vec_u_1_fxn))

        assert_estimator_accurate(estimate_delta_f, analytical_delta_f, ref_params, n_random_trials=10, atol=1e-2)


def test_endpoint_reweighting_ahfe():
    """on made-up inputs of the right shape,
    check that derivative of an absolute hydration free energy w.r.t .ligand nonbonded parameters can be computed using
    custom ops
    """
    onp.random.seed(2022)

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
    params_prime = ref_params + 0.01 * onp.random.randn(*ref_params.shape)
    v_prime, g_prime = value_and_grad(estimate_delta_f)(params_prime)
    assert v_prime != v
    assert (g_prime != g).any()
    assert np.isfinite(v_prime)
    assert np.isfinite(g_prime).all()


def test_mixture_reweighting_ahfe():
    """on made-up inputs of the right shape,
    check that derivative of an absolute hydration free energy w.r.t .ligand nonbonded parameters can be computed using
    custom ops
    """
    onp.random.seed(2022)

    _samples_0, _samples_1, batched_u_0, batched_u_1, ref_params, ref_delta_f = make_ahfe_test_system()
    fake_samples = _samples_1 + _samples_1
    fake_log_weights = onp.random.randn(len(fake_samples))

    estimate_delta_f = construct_mixture_reweighting_estimator(fake_samples, fake_log_weights, batched_u_0, batched_u_1)

    v, g = value_and_grad(estimate_delta_f)(ref_params)

    assert np.isfinite(v)
    # assert v == ref_delta_f  # not expected in this case, due to non-physical fake_log_weights
    assert np.isfinite(g).all()
    assert (g != 0).any()
    assert g.shape == ref_params.shape
    # assert anything_about_direction_of_g  # not expected because the inputs are non-physical

    # expect different estimate when evaluated on slightly different parameters
    params_prime = ref_params + 0.01 * onp.random.randn(*ref_params.shape)
    v_prime, g_prime = value_and_grad(estimate_delta_f)(params_prime)
    assert v_prime != v
    assert (g_prime != g).any()
    assert np.isfinite(v_prime)
    assert np.isfinite(g_prime).all()


def test_one_sided_exp():
    """assert consistency with pymbar.EXP on random instances"""

    onp.random.seed(2022)
    num_instances = 100

    for _ in range(num_instances):
        # instance parameters
        num_works = onp.random.randint(1, 100)
        mean = onp.random.randn() * 10
        stddev = np.exp(onp.random.randn())

        # random instance
        reduced_works = onp.random.randn(num_works) * stddev + mean

        # compare estimates
        pymbar_estimate, _ = pymbar.EXP(reduced_works)
        tm_estimate = one_sided_exp(reduced_works)

        assert np.isclose(tm_estimate, pymbar_estimate)
