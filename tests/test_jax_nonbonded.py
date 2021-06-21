import jax
jax.config.update("jax_enable_x64", True)

import numpy as onp
from numpy.random import randn, rand, randint, seed
seed(2021)

from jax import numpy as np, value_and_grad, jit
from timemachine.potentials.nonbonded import nonbonded_v3, _nonbonded_v3_clone

from functools import partial
from typing import Tuple, Callable

Conf = Params = Box = ChargeMask = LJMask = LambdaPlaneIdxs = LambdaOffsetIdxs = np.array
Lamb = Beta = Cutoff = Energy = float

NonbondedArgs = Tuple[Conf, Params, Box, Lamb, ChargeMask, LJMask, Beta, Cutoff, LambdaPlaneIdxs, LambdaOffsetIdxs]
NonbondedFxn = Callable[[*NonbondedArgs], Energy]


def generate_random_inputs(n_atoms: int, dim: int = 3) -> NonbondedArgs:
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

    args = conf, params, box, lamb, charge_rescale_mask, lj_rescale_mask, beta, cutoff, lambda_plane_idxs, lambda_offset_idxs

    return args


def compare_two_potentials(u_a: NonbondedFxn, u_b: NonbondedFxn, args: NonbondedArgs, differentiate_wrt=(0, 1, 3)):
    """Assert that energies and derivatives w.r.t. request argnums are close"""
    value_and_grads = partial(value_and_grad, argnums=differentiate_wrt)
    energy_a, gradients_a = value_and_grads(u_a)(*args)
    energy_b, gradients_b = value_and_grads(u_b)(*args)

    assert energy_a == energy_b
    onp.testing.assert_almost_equal(energy_a, energy_b)
    for (g_a, g_b) in zip(gradients_a, gradients_b):
        onp.testing.assert_allclose(g_a, g_b)


def test_jax_nonbonded(n_instances=10):
    """Assert that nonbonded_v3 and _nonbonded_v3 agree on several random instances"""
    u_a, u_b = nonbonded_v3, _nonbonded_v3_clone
    jit_u_a, jit_u_b = jit(u_a), jit(u_b)

    min_size, max_size = 10, 50

    random_sizes = onp.random.randint(min_size, max_size, n_instances)
    dims = onp.random.randint(3, 5, n_instances)

    for n_atoms, dim in zip(random_sizes, dims):
        args = generate_random_inputs(n_atoms, dim)

        compare_two_potentials(u_a, u_b, args)
        compare_two_potentials(jit_u_a, jit_u_b, args)
