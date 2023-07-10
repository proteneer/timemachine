import numpy as np
import pytest
from common import GradientTest, prepare_nb_system, prepare_water_system

from timemachine.integrator import FIXED_TO_FLOAT
from timemachine.potentials import NonbondedAllPairs, NonbondedPairListNegated

pytestmark = [pytest.mark.memcheck]


@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
def test_energy_overflow_due_to_clashes(precision, rtol, atol):
    """This test validates that if two particles overlap, the Nonbonded potential will return an inf energy"""
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
    _, _, test_energy = gpu_unbound.execute_selective(x0, params, box, False, False, True)
    np.testing.assert_allclose(reference_energy, test_energy, rtol=rtol, atol=atol)

    # Verify that there was no overflow
    _, overflows = gpu_bound.execute_fixed(x0, box)
    assert overflows == 0

    # Set the two particles to be overlapping
    x0[0, :] = x0[1, :]

    # Note that the reference returns a nan, while the unbound returns an inf
    reference_energy = potential(x0, params, box)
    assert not np.isfinite(reference_energy)
    _, _, test_energy = gpu_unbound.execute_selective(x0, params, box, False, False, True)
    assert not np.isfinite(test_energy)

    # Verify that there was an overflow that wasn't cancelled out
    _, overflows = gpu_bound.execute_fixed(x0, box)
    assert overflows != 0


@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
def test_energy_overflow_cancelled_by_exclusions(precision, rtol, atol):
    """This test validates that if two particles overlap, the Nonbonded potential will return an inf energy"""
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
    pair_list = NonbondedPairListNegated(
        potential.exclusion_idxs, potential.scale_factors, potential.beta, potential.cutoff
    )

    def compute_potential_energy(pot):
        """Verify that all the different ways of computing the energy agree"""
        gpu_pot = pot.to_gpu(precision)
        unbound_pot = gpu_pot.unbound_impl
        bound_pot = gpu_pot.bind(params).bound_impl
        _, _, selective_energy = unbound_pot.execute_selective(x0, params, box, False, False, True)
        _, bound_energy = bound_pot.execute(x0, box)
        assert selective_energy == bound_energy

        # Check that all of the energies out of a batch are identical
        _, _, energies = unbound_pot.execute_selective_batch([x0] * 2, [params] * 2, [box] * 2, False, False, True)
        for energy in energies.reshape(-1):
            assert selective_energy == energy

        fixed_energy, overflows = bound_pot.execute_fixed(x0, box)

        # If there are no overflows, the fixed energy value will match
        if overflows == 0:
            assert FIXED_TO_FLOAT(fixed_energy) == selective_energy

        return bound_energy

    reference_energy = potential(x0, params, box)
    test_energy = compute_potential_energy(potential)
    np.testing.assert_allclose(reference_energy, test_energy, rtol=rtol, atol=atol)

    # If we call the two potentials separately, we will now see that both return inf energies
    all_pairs_energy = compute_potential_energy(all_pairs)
    assert not np.isfinite(all_pairs_energy)

    pair_list_energy = compute_potential_energy(pair_list)
    assert not np.isfinite(pair_list_energy)

    # Make sure that the overflows are actually cancelling out
    pair_list_bound = pair_list.to_gpu(precision).bind(params).bound_impl
    all_pairs_bound = all_pairs.to_gpu(precision).bind(params).bound_impl

    fixed_all_pairs_energy, all_pairs_overflows = all_pairs_bound.execute_fixed(x0, box)
    fixed_pair_list_energy, pair_list_overflows = pair_list_bound.execute_fixed(x0, box)
    assert all_pairs_overflows != 0
    assert pair_list_overflows != 0
    assert all_pairs_overflows + pair_list_overflows == 0

    # The sum of the fixed point energies should be bitwise identical to the test energy
    assert FIXED_TO_FLOAT(fixed_all_pairs_energy + fixed_pair_list_energy) == test_energy
