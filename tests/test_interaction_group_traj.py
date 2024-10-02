import numpy as np
import pytest
from jax import grad, jit
from jax import numpy as jnp
from jax.scipy.special import logsumexp

from timemachine.fe.interaction_group_traj import InteractionGroupTraj, nb_pair_fxn
from timemachine.potentials.nonbonded import nonbonded_interaction_groups

pytestmark = [pytest.mark.nocuda]


def sample_random_instance(n_frames, n_env, n_lig, box_size=10, seed=2024):
    """generate random trajectory, nb_params, and decomposition into lig vs. env idxs"""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, box_size, (n_frames, n_env + n_lig, 3))

    env_idxs = np.arange(n_env)
    lig_idxs = np.arange(n_lig) + n_env
    xs[:, lig_idxs] /= box_size

    box_diags = np.ones((n_frames, 3)) * box_size

    nb_params = rng.uniform(0, 1, (n_env + n_lig, 4))
    nb_params[:, 0] = 5 * rng.normal(n_env + n_lig)  # q
    nb_params[:, 1] += 1  # sig
    nb_params[:, 2] = (5 * nb_params[:, 1]) + 1  # eps
    nb_params[:, 3] *= 1.2  # w_offset between 0 and cutoff

    return xs, box_diags, env_idxs, lig_idxs, nb_params


def test_interaction_group_traj_jax_transformable():
    """check compatibility with jit, grad"""
    config = dict(n_frames=2000, n_env=10_000, n_lig=100, box_size=10, seed=2024)
    xs, box_diags, env_idxs, lig_idxs, nb_params = sample_random_instance(**config)

    traj = InteractionGroupTraj(xs, box_diags, lig_idxs, env_idxs)

    compute_Us = traj.make_U_fxn(nb_pair_fxn)
    U_0 = compute_Us(nb_params)

    def f(params):
        Us = compute_Us(params)
        return -logsumexp(-(Us - U_0))

    grad_f = jit(grad(f))
    val = f(nb_params * 1.1)
    g = grad_f(nb_params * 1.1)
    assert np.isfinite(val)
    assert g.shape == nb_params.shape
    assert np.isfinite(g).all()


def test_interaction_group_traj_roundtrip_to_disk():
    """assert energies are consistent after round-trip to/from .npz"""
    config = dict(n_frames=2000, n_env=10_000, n_lig=100, box_size=10, seed=2024)
    xs, box_diags, env_idxs, lig_idxs, nb_params = sample_random_instance(**config)

    traj = InteractionGroupTraj(xs, box_diags, lig_idxs, env_idxs)
    compute_Us = traj.make_U_fxn(nb_pair_fxn)
    U_0 = compute_Us(nb_params)

    fname = "test_ig_traj.npz"
    traj.to_npz(fname)
    traj_2 = InteractionGroupTraj.from_npz(fname)
    compute_Us_2 = traj_2.make_U_fxn(nb_pair_fxn)
    U_0_2 = compute_Us_2(nb_params)
    np.testing.assert_allclose(U_0, U_0_2)


@pytest.mark.parametrize("config", [(50, 100, 10), (100, 100, 100), (500, 1000, 100)])
def test_interaction_group_traj_correctness(config):
    """assert energies are numerically close to jax reference nonbonded_interaction_groups implementation"""
    n_frames, n_env, n_lig = config
    target_density = 10.0
    box_size = np.cbrt((n_env + n_lig) / target_density)
    xs, box_diags, env_idxs, lig_idxs, nb_params = sample_random_instance(n_frames, n_env, n_lig, box_size=box_size)
    traj = InteractionGroupTraj(xs, box_diags, lig_idxs, env_idxs)
    compute_Us = traj.make_U_fxn(nb_pair_fxn)
    U_0 = compute_Us(nb_params)

    def U_ref(x, box_diag):
        kwargs = dict(beta=2.0, cutoff=1.2)
        vdw, es = nonbonded_interaction_groups(x, nb_params, jnp.diag(box_diag), lig_idxs, env_idxs, **kwargs)
        return jnp.sum(vdw) + jnp.sum(es)

    U_0_ref = np.array([U_ref(x, box_diag) for (x, box_diag) in zip(xs, box_diags)])
    np.testing.assert_allclose(U_0, U_0_ref)
