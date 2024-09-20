import numpy as np
import pytest
from jax import grad, jit
from jax.scipy.special import logsumexp

from timemachine.fe.interaction_group_traj import InteractionGroupTraj, nb_pair_fxn

pytestmark = [pytest.mark.nocuda]


def test_ig_traj_jax_transformable():
    """test jit(grad(f))(params), where f(params) is defined in terms of logsumexp(compute_Us(params))"""
    n_frames = 2000
    n_env = 10_000
    n_lig = 100
    box_size = 10

    np.random.seed(2024)

    xs = np.random.rand(n_frames, n_env + n_lig, 3) * box_size

    env_idxs = np.arange(n_env)
    lig_idxs = np.arange(n_lig) + n_env
    xs[:, lig_idxs] /= box_size

    box_diags = np.ones((n_frames, 3)) * box_size

    traj = InteractionGroupTraj(xs, box_diags, lig_idxs, env_idxs)

    nb_params = np.random.rand(n_env + n_lig, 4)
    nb_params[:, 0] = 5 * np.random.randn(n_env + n_lig)  # q
    nb_params[:, 1] += 1  # sig
    nb_params[:, 2] = (5 * nb_params[:, 1]) + 1  # eps
    nb_params[:, 3] *= 1.2  # w_offset between 0 and cutoff

    compute_Us = traj.make_U_fxn(nb_pair_fxn)
    U_0 = compute_Us(nb_params)

    def f(params):
        Us = compute_Us(params)
        return -logsumexp(-(Us - U_0))

    grad_f = jit(grad(f))
    _ = f(nb_params * 1.1)
    _ = grad_f(nb_params * 1.1)
