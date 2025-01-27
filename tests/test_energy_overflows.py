import jax
import numpy as np
import pytest
import scipy
from common import GradientTest, fixed_overflowed, prepare_nb_system, prepare_water_system

from timemachine.lib import custom_ops
from timemachine.lib.fixed_point import fixed_to_float
from timemachine.potentials import Nonbonded, NonbondedAllPairs, NonbondedExclusions

pytestmark = [pytest.mark.memcheck]


def verify_energies(a, b):
    """The energies can be nan, and  nan == nan = False. This uses equality except in the case of Nans for which is not equal"""
    if not (np.isnan(a) and np.isnan(b)):
        assert a == b
    else:
        assert np.isnan(a) and np.isnan(b)


@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
def test_energy_overflow_due_to_clashes(precision, rtol, atol):
    """This test validates that if two particles overlap, the Nonbonded potential will return a nan energy"""
    np.random.seed(2023)

    N = 2
    D = 3

    x0 = np.random.rand(N, D).astype(dtype=np.float64) * 2

    # Have no exclusions, only care about intermolecular energies, as exclusions should be removed
    E = 0

    box = np.eye(3) * 100.0

    params, potential = prepare_nb_system(x0, E, p_scale=1.0, cutoff=1.0)
    params[:, 3] = 0.0  # Remove 4d offset

    gpu_pot = potential.to_gpu(precision)

    gpu_unbound = gpu_pot.unbound_impl
    gpu_bound = gpu_pot.bind(params).bound_impl

    reference_energy = potential(x0, params, box)
    _, _, test_energy = gpu_unbound.execute(x0, params, box, False, False, True)
    np.testing.assert_allclose(reference_energy, test_energy, rtol=rtol, atol=atol)

    # Verify that there was no overflow before placing particles on top of each other
    fixed_energy = gpu_bound.execute_fixed(x0, box)
    assert not fixed_overflowed(fixed_energy)

    # Set the two particles to be overlapping
    x0[0, :] = x0[1, :]

    # Note that the reference potential returns an inf, while the unbound returns a nan
    reference_energy = potential(x0, params, box)
    assert not np.isfinite(reference_energy)
    _, _, test_energy = gpu_unbound.execute(x0, params, box, False, False, True)
    assert np.isnan(test_energy)

    # Verify that there was an overflow that wasn't cancelled out
    fixed_energy = gpu_bound.execute_fixed(x0, box)
    assert fixed_overflowed(fixed_energy)


@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
def test_energy_overflow_max_representation(precision, rtol, atol):
    """Validates that the GPU platform's max representation of energy is lower than that of the reference platform as the GPU
    platform only allows interaction energies that go up to MAX_LONG_LONG / FIXED_EXPONENT."""
    np.random.seed(2023)

    N = 2

    # Setup particles that are interacting from the beginning
    cutoff = 1.0
    spacing = cutoff * 0.5
    x0 = spacing * np.array([np.arange(N), np.zeros(N), np.zeros(N)]).T

    # Have no exclusions, only care about intermolecular energies, as exclusions should be removed
    E = 0

    box = np.eye(3) * 100.0

    params, potential = prepare_nb_system(x0, E, p_scale=1.0, cutoff=cutoff)
    # Set up the particles to have no LJ terms and opposite charges so they attract indefinitely
    params[0, 1] = -1  # Set first particle to have a charge of -1
    params[1, 1] = 1  # Set second particle to have a charge of 1
    params[:, 1] = 0.0  # Set sigmas to 0
    params[:, 2] = 0.0  # Set eps to 0
    params[:, 3] = 0.0  # Remove 4d offset

    bound_pot = potential.bind(params)
    gpu_pot = potential.to_gpu(precision)
    gpu_bound = gpu_pot.bind(params).bound_impl

    # Make sure that the starting system has energies that agree
    ref_u = bound_pot(x0, box)
    _, test_u = gpu_bound.execute(x0, box)
    np.testing.assert_allclose(ref_u, test_u, rtol=rtol, atol=atol)

    # Minimize the energies of the system to the point that the GPU platform stops agreeing

    ref_val_and_grad = jax.value_and_grad(bound_pot, argnums=(0,))

    def val_and_grad_fn_bfgs(x_flattened):
        x = x_flattened.reshape(x0.shape)
        u, (du_dx,) = ref_val_and_grad(x, box)
        assert np.isfinite(u), "Reference is no longer finite"
        _, gpu_u = gpu_bound.execute(x, box)
        assert not np.isnan(gpu_u) and np.abs(gpu_u) < np.iinfo(np.int64).max / custom_ops.FIXED_EXPONENT, (
            "GPU platform returned nan"
        )
        return u, du_dx.reshape(-1)

    x0_flat = x0.reshape(-1)

    method = "BFGS"

    # If we minimize the reference potential, eventually we should have the energies disagree
    with pytest.raises(AssertionError, match="GPU platform returned nan"):
        scipy.optimize.minimize(
            val_and_grad_fn_bfgs,
            x0_flat,
            method=method,
            jac=True,
        )


@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
def test_energy_overflow_negative_inf_energy(precision, rtol, atol):
    """This test validates that if two particles with +1 and -1 charges and zero'd out LJ terms, the Nonbonded potential will return a nan energy"""
    np.random.seed(2023)

    N = 2
    D = 3

    x0 = np.random.rand(N, D).astype(dtype=np.float64) * 2

    # Have no exclusions, only care about intermolecular energies, as exclusions should be removed
    E = 0

    box = np.eye(3) * 100.0

    params, potential = prepare_nb_system(x0, E, p_scale=1.0, cutoff=1.0)
    params[0, 1] = -1  # Set first particle to have a charge of -1
    params[1, 1] = 1  # Set second particle to have a charge of 1
    params[:, 1] = 0.0  # Set sigmas to 0
    params[:, 2] = 0.0  # Set eps to 0
    params[:, 3] = 0.0  # Remove 4d offset

    gpu_pot = potential.to_gpu(precision)

    gpu_unbound = gpu_pot.unbound_impl
    gpu_bound = gpu_pot.bind(params).bound_impl

    reference_energy = potential(x0, params, box)
    _, _, test_energy = gpu_unbound.execute(x0, params, box, False, False, True)
    np.testing.assert_allclose(reference_energy, test_energy, rtol=rtol, atol=atol)

    # Verify that there was no overflow before placing particles on top of each other
    fixed_energy = gpu_bound.execute_fixed(x0, box)
    assert not fixed_overflowed(fixed_energy)

    # Set the two particles to be overlapping
    x0[0, :] = x0[1, :]

    # Note that the reference potential returns -inf, while the unbound returns a nan
    reference_energy = potential(x0, params, box)
    assert not np.isfinite(reference_energy)
    _, _, test_energy = gpu_unbound.execute(x0, params, box, False, False, True)
    assert not np.isfinite(test_energy) and np.isnan(test_energy)

    # Verify that there was an overflow that wasn't cancelled out
    fixed_energy = gpu_bound.execute_fixed(x0, box)
    assert fixed_overflowed(fixed_energy)


@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
def test_energy_overflow_cancelled_by_exclusions(precision, rtol, atol):
    """This test validates that if two particles overlap, the Nonbonded potential will return a nan energy"""
    np.random.seed(2023)

    N = 9  # Multiple of three for waters
    D = 3

    x0 = GradientTest().get_water_coords(D)[:N]

    box = np.eye(3) * 100.0

    params, potential = prepare_water_system(x0, p_scale=1.0, cutoff=1.0)
    params[:, 3] = 0.0  # Remove 4d offset

    # Construct a separated version of the potential
    all_pairs = NonbondedAllPairs(
        potential.num_atoms,
        potential.beta,
        potential.cutoff,
        nblist_padding=potential.nblist_padding,
    )
    pair_list = NonbondedExclusions(potential.exclusion_idxs, potential.scale_factors, potential.beta, potential.cutoff)

    def compute_potential_energy(pot):
        """Verify that all the different ways of computing the energy agree"""
        gpu_pot = pot.to_gpu(precision)
        unbound_pot = gpu_pot.unbound_impl
        bound_pot = gpu_pot.bind(params).bound_impl
        _, _, selective_energy = unbound_pot.execute(x0, params, box, False, False, True)
        _, bound_energy = bound_pot.execute(x0, box)
        verify_energies(selective_energy, bound_energy)

        # Check that all of the energies out of a batch are identical
        _, _, energies = unbound_pot.execute_batch([x0] * 2, [params] * 2, [box] * 2, False, False, True)
        for energy in energies.reshape(-1):
            verify_energies(selective_energy, energy)

        fixed_energy = bound_pot.execute_fixed(x0, box)

        # If there are no overflows, the fixed energy value will match
        if not fixed_overflowed(fixed_energy):
            assert fixed_to_float(fixed_energy) == selective_energy

        return bound_energy

    reference_energy = potential(x0, params, box)
    test_energy = compute_potential_energy(potential)
    np.testing.assert_allclose(reference_energy, test_energy, rtol=rtol, atol=atol)

    # If we call the two potentials separately, we will now see that both return nan energies
    all_pairs_energy = compute_potential_energy(all_pairs)
    assert np.isnan(all_pairs_energy)

    pair_list_energy = compute_potential_energy(pair_list)
    assert np.isnan(pair_list_energy)

    # Make sure that the overflows are actually cancelling out
    pair_list_bound = pair_list.to_gpu(precision).bind(params).bound_impl
    all_pairs_bound = all_pairs.to_gpu(precision).bind(params).bound_impl

    fixed_all_pairs_energy = all_pairs_bound.execute_fixed(x0, box)
    fixed_pair_list_energy = pair_list_bound.execute_fixed(x0, box)
    assert fixed_overflowed(fixed_all_pairs_energy)
    assert fixed_overflowed(fixed_pair_list_energy)

    # The values are identical because of overflow, as its not clear how to go from
    # C++ __int128 to python integer.
    assert fixed_all_pairs_energy == fixed_pair_list_energy


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_energy_overflows_with_summation_of_energies(precision):
    num_atoms = 1000

    # all particles evenly spaced along x-axis, in a very large box
    spacing = 0.75
    x = spacing * np.array([np.arange(num_atoms), np.zeros(num_atoms), np.zeros(num_atoms)]).T
    box = np.eye(3) * 10000.0
    assert (np.diag(box) > spacing * num_atoms).all()

    nb_params = np.ones((num_atoms, 4))
    nb_params[:, -1] = 0.0

    nonbonded = Nonbonded(
        num_atoms=num_atoms,
        exclusion_idxs=np.array([(0, num_atoms - 1)], dtype=np.int32),
        scale_factors=np.zeros((1, 2)),
        beta=2.0,
        cutoff=1.2,
    )
    nonbonded_gpu = nonbonded.to_gpu(precision)

    # The reference should return a value larger than we can express in fixed point
    assert nonbonded(x, nb_params, box) > np.iinfo(np.int64).max / custom_ops.FIXED_EXPONENT

    # The GPU potentials will overflow, resulting in a nan value
    assert np.isnan(nonbonded_gpu(x, nb_params, box))


@pytest.mark.parametrize("size", [1, 32, 257, 1000, 10000])
def test_energy_accumulation(size):
    """Test the the logic used to accumulate energy in int128.

    It relies on doing a block level parallel reduce for performance.
    """
    rng = np.random.default_rng(2023)

    vals = rng.integers(-10000, 10000, size=size, dtype=np.int64)

    result = custom_ops._accumulate_energy(vals)

    np.testing.assert_equal(np.sum(vals), result)
