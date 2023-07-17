import numpy as np
from numpy.random import rand, randint, randn, seed

seed(2021)

from functools import partial
from typing import Any, Callable, Tuple

import pytest
from jax import jit
from jax import numpy as jnp
from jax import value_and_grad, vmap
from jax.experimental import disable_x64, enable_x64
from scipy.optimize import minimize

from timemachine.constants import BOLTZ, DEFAULT_TEMP
from timemachine.fe.reweighting import one_sided_exp
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.md import builders
from timemachine.potentials.jax_utils import get_all_pairs_indices, pairs_from_interaction_groups, pairwise_distances
from timemachine.potentials.nonbonded import (
    basis_expand_lj_atom,
    basis_expand_lj_env,
    convert_exclusions_to_rescale_masks,
    coulomb_interaction_group_energy,
    coulomb_prefactors_on_traj,
    direct_space_pme,
    lennard_jones,
    lj_interaction_group_energy,
    lj_prefactors_on_traj,
    nonbonded,
    nonbonded_block,
    nonbonded_on_specific_pairs,
    nonbonded_pair,
)

Array = Any
Conf = Array
Params = Array
Box = Array
ExclusionIdxs = Array
ScaleFactors = Array
Beta = float
Cutoff = float
Energy = float

NonbondedArgs = Tuple[Conf, Params, Box, ExclusionIdxs, ScaleFactors, Beta, Cutoff]
NonbondedFxn = Callable[[Conf, Params, Box, ExclusionIdxs, ScaleFactors, Beta, Cutoff], Energy]

pytestmark = [pytest.mark.nogpu]


def resolve_clashes(x0, box0, min_dist=0.1):
    def urt(x, box):
        distance_matrix = pairwise_distances(x, box)
        i, j = np.triu_indices(len(distance_matrix), k=1)
        return distance_matrix[i, j]

    dij = urt(x0, box0)
    x_shape = x0.shape
    box_shape = box0.shape

    if jnp.min(dij) < min_dist:
        # print('some distances too small')
        print(f"before optimization: min(dij) = {jnp.min(dij)} < min_dist threshold ({min_dist})")
        # print('smallest few distances', sorted(dij)[:10])

        def unflatten(xbox):
            n = x_shape[0] * x_shape[1]
            x = xbox[:n].reshape(x_shape)
            box = xbox[n:].reshape(box_shape)
            return x, box

        def U_repulse(xbox):
            x, box = unflatten(xbox)
            dij = urt(x, box)
            return jnp.sum(jnp.where(dij < min_dist, (dij - min_dist) ** 2, 0))

        def fun(xbox):
            v, g = value_and_grad(U_repulse)(xbox)
            return float(v), np.array(g, np.float64)

        initial_state = jnp.hstack([x0.flatten(), box0.flatten()])
        # print(f'penalty before: {U_repulse(initial_state)}')
        result = minimize(fun, initial_state, jac=True, method="L-BFGS-B")
        # print(f'penalty after minimization: {U_repulse(result.x)}')

        x, box = unflatten(result.x)
        dij = urt(x, box)

        print(f"after optimization: min(dij) = {jnp.min(dij)}")

        return x, box

    else:
        return x0, box0


easy_instance_flags = dict(
    trigger_pbc=False,
    randomize_box=False,
    randomize_charges=False,
    randomize_sigma=False,
    randomize_epsilon=False,
    randomize_charge_rescale_mask=False,
    randomize_lj_rescale_mask=False,
    randomize_w_coords=False,
    randomize_beta=False,
    randomize_cutoff=False,
)

difficult_instance_flags = {key: True for key in easy_instance_flags}


def generate_waterbox_nb_args() -> NonbondedArgs:
    ff = Forcefield.load_default()
    system, conf, box, _ = builders.build_water_system(3.0, ff.water_ff)
    bps, masses = openmm_deserializer.deserialize_system(system, cutoff=1.2)
    nb = bps[-1]
    params = nb.params

    beta = nb.potential.beta
    cutoff = nb.potential.cutoff

    exclusion_idxs = np.zeros((0,), dtype=np.int32)
    scale_factors = np.zeros((0, 2))

    args = (
        conf,
        params,
        box,
        exclusion_idxs,
        scale_factors,
        beta,
        cutoff,
    )

    return args


def generate_random_inputs(n_atoms, dim, instance_flags=difficult_instance_flags) -> NonbondedArgs:
    """Can toggle randomization of each argument using instance_flags"""
    box = jnp.eye(dim)
    if instance_flags["randomize_box"]:
        box += jnp.diag(rand(dim))
    assert box.shape == (dim, dim)

    conf = rand(n_atoms, dim)
    if instance_flags["trigger_pbc"]:
        conf *= 5
        conf -= 2.5

    min_dist = 0.1
    conf, box = resolve_clashes(conf, box, min_dist=min_dist)
    box = np.array(box)

    cutoff = 1.2
    if instance_flags["randomize_cutoff"]:
        cutoff += rand()

    charges = jnp.zeros(n_atoms)
    sig = min_dist * jnp.ones(n_atoms)
    eps = jnp.ones(n_atoms)
    w_coords = jnp.zeros(n_atoms)
    if instance_flags["randomize_charges"]:
        charges = jnp.array(randn(n_atoms))
    if instance_flags["randomize_sigma"]:
        sig = min_dist * jnp.array(rand(n_atoms))
    if instance_flags["randomize_epsilon"]:
        eps = jnp.array(rand(n_atoms))
    if instance_flags["randomize_w_coords"]:
        w_coords = 2 * cutoff * (2 * jnp.array(rand(n_atoms)) - 1)  # [-2 * cutoff, 2 * cutoff]

    params = jnp.array([charges, sig, eps, w_coords]).T

    exclusion_idxs_ = []
    scale_factors_ = []

    for _ in range(n_atoms):
        i, j = randint(n_atoms, size=2)
        exclusion_idxs_.append((i, j))
        charge_scale_factor = 1.0
        lj_scale_factor = 1.0
        if instance_flags["randomize_charge_rescale_mask"]:
            charge_scale_factor = 0.0
        if instance_flags["randomize_lj_rescale_mask"]:
            lj_scale_factor = 0.0
        scale_factors_.append((charge_scale_factor, lj_scale_factor))

    beta = 2.0
    if instance_flags["randomize_beta"]:
        beta += rand()

    exclusion_idxs = np.array(exclusion_idxs_, dtype=np.int32)
    scale_factors = np.array(scale_factors_)
    args = (conf, params, box, exclusion_idxs, scale_factors, beta, cutoff)

    return args


def compare_two_potentials(u_a: NonbondedFxn, u_b: NonbondedFxn, args: NonbondedArgs, differentiate_wrt=(0, 1)):
    """Assert that energies and derivatives w.r.t. request argnums are close"""
    value_and_grads = partial(value_and_grad, argnums=differentiate_wrt)
    energy_a, gradients_a = value_and_grads(u_a)(*args)
    energy_b, gradients_b = value_and_grads(u_b)(*args)

    np.testing.assert_allclose(energy_a, energy_b)
    for (g_a, g_b) in zip(gradients_a, gradients_b):
        np.testing.assert_allclose(g_a, g_b)


def _nonbonded_clone(
    conf,
    params,
    box,
    exclusion_idxs,
    scale_factors,
    beta,
    cutoff,
):
    """See docstring of `nonbonded` for more details

    This is here just for testing purposes, to mimic the signature of `nonbonded` but to use
    `nonbonded_on_specific_pairs` under the hood.
    """

    N = conf.shape[0]
    charge_rescale_mask, lj_rescale_mask = convert_exclusions_to_rescale_masks(exclusion_idxs, scale_factors, N)

    # TODO: len(pairs) == n_interactions -- may want to break this
    #   up into more manageable blocks if n_interactions is large
    pairs = get_all_pairs_indices(N)

    lj, coulomb = nonbonded_on_specific_pairs(conf, params, box, pairs, beta, cutoff)

    # keep only eps > 0
    inds_i, inds_j = pairs.T
    eps = params[:, 2]
    lj = jnp.where(eps[inds_i] > 0, lj, 0)
    lj = jnp.where(eps[inds_j] > 0, lj, 0)

    eij_total = lj * lj_rescale_mask[inds_i, inds_j] + coulomb * charge_rescale_mask[inds_i, inds_j]

    return jnp.sum(eij_total)


def run_randomized_tests_of_jax_nonbonded(instance_generator, n_instances=10):
    """Assert that `nonbonded` and `_nonbonded_clone` agree on several random instances

    instance_generator(n_atoms, dim) -> NonbondedArgs
    """

    min_size, max_size = 10, 50

    random_sizes = np.random.randint(min_size, max_size, n_instances)
    dims = np.random.randint(3, 5, n_instances)

    for n_atoms, dim in zip(random_sizes, dims):
        args = instance_generator(n_atoms, dim)
        conf, params, box, exclusion_idxs, scale_factors, beta, cutoff = args
        diff_args = conf, params, box, beta, cutoff

        # Need to differentiate between the args that can be traced
        # and the args that are fixed. Otherwise we get a
        # TracerArrayConversionError for exclusion_idxs and scale_factors.
        def jittable_nonbonded(conf, params, box, beta, cutoff):
            return nonbonded(conf, params, box, exclusion_idxs, scale_factors, beta, cutoff, runtime_validate=False)

        def jittable_nonbonded_clone(conf, params, box, beta, cutoff):
            return _nonbonded_clone(conf, params, box, exclusion_idxs, scale_factors, beta, cutoff)

        u_a = jit(jittable_nonbonded)
        u_b = jit(jittable_nonbonded_clone)

        compare_two_potentials(u_a, u_b, diff_args)


def test_jax_nonbonded_waterbox():
    jittable_nonbonded = partial(nonbonded, runtime_validate=False)
    u_a, u_b = jit(jittable_nonbonded), jit(_nonbonded_clone)
    compare_two_potentials(u_a, u_b, generate_waterbox_nb_args())


def test_jax_nonbonded_easy(n_instances=10):
    instance_generator = partial(generate_random_inputs, instance_flags=easy_instance_flags)
    run_randomized_tests_of_jax_nonbonded(instance_generator, n_instances)


def test_jax_nonbonded(n_instances=10):
    instance_generator = partial(generate_random_inputs, instance_flags=difficult_instance_flags)
    run_randomized_tests_of_jax_nonbonded(instance_generator, n_instances)


def test_vmap():
    """Can call jit(vmap(nonbonded_on_specific_pairs))"""

    # # atoms in "ligand" vs. "environment"
    n_ligand, n_environment = 50, 1000
    n_total = n_ligand + n_environment
    conf, params, box, _, _, beta, cutoff = generate_random_inputs(n_total, 3)

    ligand_indices = np.arange(n_ligand)
    environment_indices = np.arange(n_environment) + n_ligand
    pairs = pairs_from_interaction_groups(ligand_indices, environment_indices)

    n_interactions = len(pairs)

    fixed_kwargs = dict(params=params, box=box, pairs=pairs, beta=beta, cutoff=cutoff)

    # signature: conf -> ljs, coulombs, where ljs.shape == (n_interactions, )
    u_pairs = partial(nonbonded_on_specific_pairs, **fixed_kwargs)
    ljs, coulombs = u_pairs(conf)
    assert ljs.shape == (n_interactions,)

    def u(conf):
        ljs, coulombs = u_pairs(conf)
        return jnp.sum(ljs + coulombs)

    # vmap over snapshots
    vmapped = jit(vmap(u))
    n_snapshots = 100
    confs = np.random.randn(n_snapshots, n_total, 3)
    us = vmapped(confs)
    assert us.shape == (n_snapshots,)


def test_jax_nonbonded_block():
    """Assert that nonbonded_block and nonbonded_on_specific_pairs agree"""
    ff = Forcefield.load_default()
    system, conf, box, _ = builders.build_water_system(3.0, ff.water_ff)
    bps, masses = openmm_deserializer.deserialize_system(system, cutoff=1.2)
    nb = bps[-1]
    params = nb.params

    N = conf.shape[0]
    beta = nb.potential.beta
    cutoff = nb.potential.cutoff

    split = 70

    def u_a(x, box, params):
        xi = x[:split]
        xj = x[split:]
        pi = params[:split]
        pj = params[split:]
        return nonbonded_block(xi, xj, box, pi, pj, beta, cutoff)

    i_s, j_s = jnp.indices((split, N - split))
    indices_left = i_s.flatten()
    indices_right = j_s.flatten() + split
    pairs = jnp.array([indices_left, indices_right]).T

    def u_b(x, box, params):
        vdw, es = nonbonded_on_specific_pairs(x, params, box, pairs, beta, cutoff)

        return jnp.sum(vdw + es)

    np.testing.assert_allclose(u_a(conf, box, params), u_b(conf, box, params))


def test_lj_basis():
    """Randomized test that LJ(sig, eps) computed via basis expansion matches reference"""

    np.random.seed(2023)

    n_env = 10_000

    r_i = 5 * np.random.rand(n_env)  # r_i > 0, should exercise both (r_i << sig + sig_i) and (r >> sig + sig_i)

    # other particles have random params
    sig_i = np.random.rand(n_env)
    eps_i = np.random.rand(n_env)

    def lj_ref(sig, eps):
        return np.sum(lennard_jones(r_i, sig_i + sig, eps_i * eps))

    lj_prefactors = basis_expand_lj_env(sig_i, eps_i, r_i)

    def lj_basis(sig, eps):
        projection = basis_expand_lj_atom(sig, eps)
        return jnp.dot(projection, lj_prefactors)

    for _ in range(100):
        sig = np.random.rand()
        eps = np.random.rand()

        u_ref = lj_ref(sig, eps)
        u_test = lj_basis(sig, eps)

        np.testing.assert_allclose(u_test, u_ref)


def test_precomputation():
    """Assert that nonbonded interaction groups using precomputation agree with reference nonbonded_on_specific_pairs"""
    ff = Forcefield.load_default()
    system, conf, box, _ = builders.build_water_system(3.0, ff.water_ff)
    bps, masses = openmm_deserializer.deserialize_system(system, cutoff=1.2)
    nb = bps[-1]
    params = nb.params

    n_atoms = conf.shape[0]
    beta = nb.potential.beta
    cutoff = nb.potential.cutoff

    # generate array in the shape of a "trajectory" by adding noise to an initial conformation
    n_snapshots = 100
    np.random.seed(2022)
    traj = jnp.array([conf] * n_snapshots) + np.random.randn(n_snapshots, *conf.shape) * 0.005
    boxes = jnp.array([box] * n_snapshots) * (1 - 0.0025 + np.random.rand(n_snapshots, *box.shape) * 0.005)

    # split system into "ligand" vs. "environment"
    n_ligand = 3 * 10  # call the first 10 waters in the system "ligand" and the rest "environment"
    ligand_idx = np.arange(n_ligand)
    env_idx = np.arange(n_ligand, n_atoms)
    pairs = pairs_from_interaction_groups(ligand_idx, env_idx)

    # reference version: nonbonded_on_specific_pairs
    def u_ref(x, box, params):
        vdw, es = nonbonded_on_specific_pairs(x, params, box, pairs, beta, cutoff)
        return jnp.sum(vdw + es)

    @jit
    def u_batch_ref(sig_ligand, eps_ligand, q_ligand):
        new_params = jnp.array(params).at[ligand_idx, 0].set(q_ligand)
        new_params = new_params.at[ligand_idx, 1].set(sig_ligand)
        new_params = new_params.at[ligand_idx, 2].set(eps_ligand)

        def f(x, box):
            return u_ref(x, box, new_params)

        return vmap(f)(traj, boxes)

    # test version: with precomputation
    charges, sigmas, epsilons, _ = params.T
    lj_prefactors = lj_prefactors_on_traj(traj, boxes, sigmas, epsilons, ligand_idx, env_idx, cutoff)
    q_prefactors = coulomb_prefactors_on_traj(traj, boxes, charges, ligand_idx, env_idx, beta, cutoff)

    @jit
    def u_batch_test(sig_ligand, eps_ligand, q_ligand):
        vdw = vmap(lj_interaction_group_energy, (None, None, 0))(sig_ligand, eps_ligand, lj_prefactors)
        es = coulomb_interaction_group_energy(q_ligand, q_prefactors)
        return vdw + es

    # generate many sets of ligand parameters to test on
    sig_ligand_0 = sigmas[ligand_idx]
    eps_ligand_0 = epsilons[ligand_idx]
    q_ligand_0 = charges[ligand_idx]

    temperature = DEFAULT_TEMP
    kBT = BOLTZ * temperature

    def make_reweighter(u_batch_fxn):
        u_0 = u_batch_fxn(sig_ligand_0, eps_ligand_0, q_ligand_0)

        def reweight(sig_ligand, eps_ligand, q_ligand):
            delta_us = (u_batch_fxn(sig_ligand, eps_ligand, q_ligand) - u_0) / kBT
            return one_sided_exp(delta_us)

        return reweight

    reweight_ref = jit(make_reweighter(u_batch_ref))
    reweight_test = jit(make_reweighter(u_batch_test))

    for _ in range(5):
        # abs() so sig, eps will be non-negative
        sig_ligand = jnp.abs(sig_ligand_0 + (0.2 * np.random.randn(n_ligand) - 0.1))
        eps_ligand = jnp.abs(eps_ligand_0 + (0.2 * np.random.rand(n_ligand) - 0.1))
        q_ligand = q_ligand_0 + np.random.randn(n_ligand)

        expected = u_batch_ref(sig_ligand, eps_ligand, q_ligand)
        actual = u_batch_test(sig_ligand, eps_ligand, q_ligand)

        # test array of energies is ~equal to reference
        np.testing.assert_allclose(actual, expected)

        # test that reweighting estimates and gradients are ~equal to reference
        v_ref, gs_ref = value_and_grad(reweight_ref, argnums=(0, 1, 2))(sig_ligand, eps_ligand, q_ligand)
        v_test, gs_test = value_and_grad(reweight_test, argnums=(0, 1, 2))(sig_ligand, eps_ligand, q_ligand)

        np.testing.assert_allclose(v_ref, v_test)
        np.testing.assert_allclose(gs_ref, gs_test)


@pytest.mark.parametrize("distance", [0.0, 0.00000001, 0.00001, 0.001, np.inf])  # include 0
@pytest.mark.parametrize("jax_precision_mode", [disable_x64, enable_x64])
@pytest.mark.parametrize("potential", ["lj", "es", "combined"])
def test_nb_pair_not_nan(distance, jax_precision_mode, potential):
    qij = -1.0
    sig = 0.3
    eps = 0.1
    beta = 2.0
    cutoff = 1.2

    def U_lj(r):
        return lennard_jones(jnp.array(r), sig, eps)

    def U_es(r):
        return direct_space_pme(jnp.array(r), qij, beta)

    def U_combined(r):
        return nonbonded_pair(jnp.array(r), qij, sig, eps, beta, cutoff)

    potentials = {"lj": U_lj, "es": U_es, "combined": U_combined}
    U_fn = potentials[potential]

    with jax_precision_mode():
        nrg = U_fn(distance)
        assert not np.isnan(nrg)
        assert nrg > -np.inf


@pytest.mark.parametrize("jax_precision_mode", [disable_x64, enable_x64])
def test_nonbonded_block_zero_distance(jax_precision_mode):
    """check case of collocated oppositely charged particles"""
    rng = np.random.default_rng(1234)

    n_particles = 50
    xi = rng.normal(size=(n_particles, 3))
    xj = np.array(xi)  # duplicate positions, so at least some d_ij == 0

    # random parameters: (sig > 0, eps > 0, charges can be > 0 or < 0)
    params_i = rng.normal(size=(n_particles, 3))
    params_i[:, :2] = np.abs(params_i[:, :2])

    params_j = rng.normal(size=(n_particles, 3))
    params_j[:, :2] = np.abs(params_j[:, :2])

    beta = 2.0
    cutoff = 1.2

    q_i = params_i[:, 2]
    q_j = params_j[:, 2]
    assert np.sign(q_i) != np.sign(q_j).any(), "failed to generate informative random test"

    box = np.eye(3) * 5

    with jax_precision_mode():
        U = nonbonded_block(xi, xj, box, params_i, params_j, beta, cutoff)
        assert not np.isnan(U)
        assert U > -np.inf
