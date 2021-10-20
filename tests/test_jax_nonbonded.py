import jax
jax.config.update("jax_enable_x64", True)

import numpy as onp
from numpy.random import randn, rand, randint, seed
seed(2021)

from jax import numpy as np, value_and_grad, jit, vmap

from jax.ops import index_update, index
from timemachine.potentials.nonbonded import nonbonded_v3, nonbonded_v3_on_specific_pairs
from timemachine.potentials.jax_utils import convert_to_4d, get_all_pairs_indices, get_group_group_indices, distance

from functools import partial
from typing import Tuple, Callable

Conf = Params = Box = ChargeMask = LJMask = LambdaPlaneIdxs = LambdaOffsetIdxs = np.array
Lamb = Beta = Cutoff = Energy = float

nonbonded_args = Conf, Params, Box, Lamb, ChargeMask, LJMask, Beta, Cutoff, LambdaPlaneIdxs, LambdaOffsetIdxs
NonbondedArgs = Tuple[nonbonded_args]
NonbondedFxn = Callable[[*nonbonded_args], Energy]


def resolve_clashes(x0, box0, min_dist=0.1):
    def urt(x, box):
        distance_matrix = distance(x, box)
        i, j = np.triu_indices(len(distance_matrix), k=1)
        return distance_matrix[i, j]

    dij = urt(x0, box0)
    x_shape = x0.shape
    box_shape = box0.shape

    if np.min(dij) < min_dist:
        #print('some distances too small')
        print(f'before optimization: min(dij) = {np.min(dij)} < min_dist threshold ({min_dist})')
        #print('smallest few distances', sorted(dij)[:10])

        from scipy.optimize import minimize
        def unflatten(xbox):
            n = x_shape[0] * x_shape[1]
            x = xbox[:n].reshape(x_shape)
            box = xbox[n:].reshape(box_shape)
            return x, box

        def U_repulse(xbox):
            x, box = unflatten(xbox)
            dij = urt(x, box)
            return np.sum(np.where(dij < min_dist, (dij - min_dist)**2, 0))

        def fun(xbox):
            v, g = value_and_grad(U_repulse)(xbox)
            return float(v), onp.array(g, onp.float64)

        initial_state = np.hstack([x0.flatten(), box0.flatten()])
        #print(f'penalty before: {U_repulse(initial_state)}')
        result = minimize(fun, initial_state, jac=True, method='L-BFGS-B')
        #print(f'penalty after minimization: {U_repulse(result.x)}')


        x, box = unflatten(result.x)
        dij = urt(x, box)

        print(f'after optimization: min(dij) = {np.min(dij)}')

        return x, box

    else:
        return x0, box0


def generate_easy_random_inputs(n_atoms, dim) -> NonbondedArgs:
    box = np.eye(dim)
    assert box.shape == (dim, dim)

    conf = rand(n_atoms, dim)

    conf, box = resolve_clashes(conf, box, min_dist=0.1)

    charges = np.zeros(n_atoms)
    sig = 0.2 * np.ones(n_atoms)
    eps = 1.0 * np.ones(n_atoms)
    params = np.array([charges, sig, eps]).T

    lamb = 0.5
    charge_rescale_mask = onp.ones((n_atoms, n_atoms))
    lj_rescale_mask = onp.ones((n_atoms, n_atoms))

    beta = 1.2
    cutoff = 1.0

    lambda_plane_idxs = np.zeros(n_atoms, dtype=int)
    lambda_offset_idxs = np.zeros(n_atoms, dtype=int)

    args = conf, params, box, lamb, charge_rescale_mask, lj_rescale_mask, beta, cutoff, lambda_plane_idxs, lambda_offset_idxs

    return args



def generate_random_inputs(n_atoms: int, dim: int = 3, min_dist=0.1) -> NonbondedArgs:
    box = 1 + np.diag(rand(dim))  # each side length ~ Unif([1, 2])
    assert box.shape == (dim, dim)

    conf = rand(n_atoms, dim) * 5 - 1  # trigger periodic wrapping behavior

    conf, box = resolve_clashes(conf, box, min_dist)

    # charges ~ Normal(0,1), sigmas ~ Unif([min_dist, 1 + min_dist]), eps ~ Unif([0,1])
    charges = randn(n_atoms)
    sig = rand(n_atoms) + min_dist # sig ~= 0 would be bad
    eps = rand(n_atoms)
    params = np.array([charges, sig, eps]).T

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

    onp.testing.assert_almost_equal(energy_a, energy_b)
    for (g_a, g_b) in zip(gradients_a, gradients_b):
        onp.testing.assert_allclose(g_a, g_b)


def _nonbonded_v3_clone(
        conf,
        params,
        box,
        lamb,
        charge_rescale_mask,
        lj_rescale_mask,
        beta,
        cutoff,
        lambda_plane_idxs,
        lambda_offset_idxs,
):
    """See docstring of nonbonded_v3 for more details

    This is here just for testing purposes, to mimic the signature of nonbonded_v3 but to use
    nonbonded_v3_on_specific_pairs under the hood.
    """

    N = conf.shape[0]

    if conf.shape[-1] == 3:
        conf = convert_to_4d(conf, lamb, lambda_plane_idxs, lambda_offset_idxs, cutoff)

    # make 4th dimension of box large enough so its roughly aperiodic
    if box is not None:
        if box.shape[-1] == 3:
            box_4d = np.eye(4) * 1000
            box_4d = index_update(box_4d, index[:3, :3], box)
        else:
            box_4d = box
    else:
        box_4d = None
    box = box_4d

    # TODO: len(inds_i) == n_interactions -- may want to break this
    #   up into more manageable blocks if n_interactions is large
    inds_i, inds_j = get_all_pairs_indices(N)

    lj, coulomb = nonbonded_v3_on_specific_pairs(conf, params, box, inds_i, inds_j, beta, cutoff)

    eij_total = lj * lj_rescale_mask[inds_i, inds_j] + coulomb * charge_rescale_mask[inds_i, inds_j]

    return np.sum(eij_total)


def run_randomized_tests_of_jax_nonbonded(instance_generator=generate_random_inputs, n_instances=10):
    """Assert that nonbonded_v3 and _nonbonded_v3 agree on several random instances"""
    jittable_nonbonded_v3 = partial(nonbonded_v3, runtime_validate=False)
    u_a, u_b = jit(jittable_nonbonded_v3), jit(_nonbonded_v3_clone)

    min_size, max_size = 10, 50

    random_sizes = onp.random.randint(min_size, max_size, n_instances)
    dims = onp.random.randint(3, 5, n_instances)

    for n_atoms, dim in zip(random_sizes, dims):
        args = instance_generator(n_atoms, dim)
        compare_two_potentials(u_a, u_b, args)


def test_jax_nonbonded(n_instances=10):
    run_randomized_tests_of_jax_nonbonded(generate_random_inputs, n_instances)


def test_jax_nonbonded_easy(n_instances=10):
    run_randomized_tests_of_jax_nonbonded(generate_easy_random_inputs, n_instances)


def test_vmap():
    """Can call jit(vmap(nonbonded_v3_on_specific_pairs))"""

    # # atoms in "ligand" vs. "environment"
    n_ligand, n_environment = 50, 1000
    n_total = n_ligand + n_environment
    conf, params, box, lamb, _, _, beta, cutoff, _, _ = generate_random_inputs(n_total, 3)

    inds_i, inds_j = get_group_group_indices(n_ligand, n_environment)
    inds_j += n_ligand
    n_interactions = len(inds_i)

    fixed_kwargs = dict(params=params, box=box, inds_l=inds_i, inds_r=inds_j, beta=beta, cutoff=cutoff)

    # signature: conf -> ljs, coulombs, where ljs.shape == (n_interactions, )
    u_pairs = partial(nonbonded_v3_on_specific_pairs, **fixed_kwargs)
    ljs, coulombs = u_pairs(conf)
    assert ljs.shape == (n_interactions, )

    def u(conf):
        ljs, coulombs = u_pairs(conf)
        return np.sum(ljs + coulombs)

    # vmap over snapshots
    vmapped = jit(vmap(u))
    n_snapshots = 100
    confs = onp.random.randn(n_snapshots, n_total, 3)
    us = vmapped(confs)
    assert us.shape == (n_snapshots, )
