# Generate some random inputs!

import jax
jax.config.update("jax_enable_x64", True)

import numpy as onp
from numpy.random import randn, rand, randint
from jax import numpy as np, value_and_grad, jit
from timemachine.potentials.nonbonded import nonbonded_v3, _nonbonded_v3_clone


def generate_random_inputs(n_atoms, dim=3):
    # jittered
    offsets = onp.array([onp.arange(n_atoms)] * dim).T
    assert (offsets.shape == (n_atoms, dim))
    for i in range(dim):
        onp.random.shuffle(offsets[:, i])
    conf = (0.5 * randn(n_atoms, dim) + offsets)
    conf += conf.min()
    params = rand(n_atoms, 3)
    params[:, 0] -= np.mean(params[:, 0])

    box = np.diag(0.5 + 0.5 * rand(3)) * 10 * n_atoms

    lamb = rand()

    charge_rescale_mask = onp.ones((n_atoms, n_atoms))
    lj_rescale_mask = onp.ones((n_atoms, n_atoms))
    for _ in range(n_atoms):
        i, j = randint(n_atoms, size=2)
        charge_rescale_mask[i, j] = 0.0
        lj_rescale_mask[i, j] = 0.0

    beta = rand() + 1
    cutoff = rand() + 0.5

    lambda_plane_idxs = randint(low=-2, high=2, size=n_atoms)
    lambda_offset_idxs = randint(low=-2, high=2, size=n_atoms)

    return conf, params, box, lamb, charge_rescale_mask, lj_rescale_mask, beta, cutoff, lambda_plane_idxs, lambda_offset_idxs


def compare_two_potentials(u_a, u_b, args, differentiate_wrt=(0, 1, 3)):
    energy_a, gradients_a = value_and_grad(u_a, argnums=differentiate_wrt)(
        *args)
    energy_b, gradients_b = value_and_grad(u_b, argnums=differentiate_wrt)(
        *args)

    assert energy_a == energy_b
    onp.testing.assert_almost_equal(energy_a, energy_b)
    for (g_a, g_b) in zip(gradients_a, gradients_b):
        onp.testing.assert_allclose(g_a, g_b)


def test_jax_nonbonded():
    u_a, u_b = nonbonded_v3, _nonbonded_v3_clone
    args = generate_random_inputs(20, 3)

    compare_two_potentials(u_a, u_b, args)
    compare_two_potentials(jit(u_a), jit(u_b), args)
