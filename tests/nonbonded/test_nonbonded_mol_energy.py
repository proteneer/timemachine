import numpy as np
import pytest

from timemachine.constants import DEFAULT_TEMP
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import custom_ops
from timemachine.md import builders
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.exchange.exchange_mover import BDExchangeMove, randomly_rotate_and_translate
from timemachine.potentials import HarmonicBond, Nonbonded


@pytest.mark.memcheck
@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_nonbonded_mol_energy_potential_validation(precision):
    rng = np.random.default_rng(2023)
    ff = Forcefield.load_default()
    system, conf, box, _ = builders.build_water_system(2.0, ff.water_ff)

    N = 10
    cutoff = 1.2
    beta = 2.0

    conf = conf[:N]
    params = rng.uniform(size=(conf.shape[0], 4))

    indices_beyond_range_group_indices = [[0, 1, 2], [N + 1]]

    klass = custom_ops.NonbondedMolEnergyPotential_f32
    if precision == np.float64:
        klass = custom_ops.NonbondedMolEnergyPotential_f64
    with pytest.raises(RuntimeError, match="Grouped indices must be between 0 and N"):
        klass(N, indices_beyond_range_group_indices, beta, cutoff)

    indices_in_multiple_groups = [[0, 1, 2], [2, 3]]
    with pytest.raises(RuntimeError, match="All grouped indices must be unique"):
        klass(N, indices_in_multiple_groups, beta, cutoff)

    group_idxs = [[0, 1, 2]]
    pot = klass(N, group_idxs, beta, cutoff)
    with pytest.raises(RuntimeError, match="params N != coords N"):
        pot.execute(np.concatenate([conf] * 2), params, box)

    with pytest.raises(RuntimeError, match="params N != coords N"):
        pot.execute(conf, np.concatenate([params] * 2), box)

    with pytest.raises(RuntimeError, match="N != N_"):
        pot.execute(np.concatenate([conf] * 2), np.concatenate([conf] * 2), box)

    pot.execute(conf, params, box)


@pytest.mark.memcheck
@pytest.mark.parametrize("num_mols", [1, 2, 15, 100, 4085])
@pytest.mark.parametrize("precision,atol,rtol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 2e-3)])
def test_nonbonded_mol_energy_matches_exchange_mover_batch_U(num_mols, precision, atol, rtol):
    """Assert that NonbondedMolEnergyPotential Cuda implementation produces the same
    energies as the reference jax version in the BDExchangeMover"""
    rng = np.random.default_rng(2023)
    ff = Forcefield.load_default()
    system, conf, box, _ = builders.build_water_system(5.0, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)
    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), conf.shape[0])

    assert len(all_group_idxs) >= num_mols

    # Get the indices of atoms within each molecule
    group_idxs = all_group_idxs[:num_mols]

    # Shuffle the indices to be sure ordering doesn't matter
    rng.shuffle(group_idxs)

    conf_idxs = np.array(group_idxs).reshape(-1)
    conf = conf[conf_idxs]
    N = conf.shape[0]

    params = nb.params[conf_idxs]

    beta = nb.potential.beta
    cutoff = nb.potential.cutoff
    klass = custom_ops.NonbondedMolEnergyPotential_f32
    if precision == np.float64:
        klass = custom_ops.NonbondedMolEnergyPotential_f64

    mover = BDExchangeMove(beta, cutoff, params, group_idxs, DEFAULT_TEMP)

    mol_by_mol_pot = klass(N, group_idxs, beta, cutoff)

    def u_ref(x, box, params):
        return mover.batch_U_fn(x, box, mover.all_a_idxs, mover.all_b_idxs)

    def u_test(x, box, params):
        mol_energies = mol_by_mol_pot.execute(x, params, box)
        assert mol_energies.shape == (len(group_idxs),)

        # Make sure running again gives bitwise identical results
        comp_mol_energies = mol_by_mol_pot.execute(x, params, box)
        np.testing.assert_array_equal(mol_energies, comp_mol_energies)
        return mol_energies

    np.testing.assert_allclose(u_test(conf, box, params), u_ref(conf, box, params), rtol=rtol, atol=atol)


@pytest.mark.parametrize("num_mols", [500])
@pytest.mark.parametrize("moves", [100])
@pytest.mark.parametrize("precision,atol,rtol", [(np.float64, 1e-8, 1e-8), (np.float32, 5e-4, 2e-3)])
def test_nonbonded_mol_energy_random_moves(num_mols, moves, precision, atol, rtol):
    """Verify that with random move for waters that the exchange mover and Nonbonded water match in the case
    where clashes are likely to be introduced
    """
    rng = np.random.default_rng(2023)
    ff = Forcefield.load_default()
    system, conf, _, _ = builders.build_water_system(4.0, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)
    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), conf.shape[0])

    assert len(all_group_idxs) >= num_mols

    # Get the indices of atoms within each molecule
    group_idxs = all_group_idxs[:num_mols]

    conf_idxs = np.array(group_idxs).reshape(-1)

    conf = conf[conf_idxs]
    # Compute the box
    box_lengths = np.amax(conf, axis=0) - np.amin(conf, axis=0)
    box = np.eye(3, dtype=np.float64) * box_lengths

    N = conf.shape[0]

    params = nb.params[conf_idxs]

    beta = nb.potential.beta
    cutoff = nb.potential.cutoff
    klass = custom_ops.NonbondedMolEnergyPotential_f32
    if precision == np.float64:
        klass = custom_ops.NonbondedMolEnergyPotential_f64

    mover = BDExchangeMove(beta, cutoff, params, group_idxs, DEFAULT_TEMP)

    mol_by_mol_pot = klass(N, group_idxs, beta, cutoff)

    def u_ref(x, box, params):
        return mover.batch_U_fn(x, box, mover.all_a_idxs, mover.all_b_idxs)

    def u_test(x, box, params):
        mol_energies = mol_by_mol_pot.execute(x, params, box)
        assert mol_energies.shape == (len(group_idxs),)

        # Make sure running again gives bitwise identical results
        comp_mol_energies = mol_by_mol_pot.execute(x, params, box)
        np.testing.assert_array_equal(mol_energies, comp_mol_energies)
        return mol_energies

    np.testing.assert_allclose(u_test(conf, box, params), u_ref(conf, box, params), rtol=rtol, atol=atol)

    mols_to_move = rng.choice(np.arange(len(group_idxs)), size=moves)

    # empirical threshold based on testing
    threshold = 1e8

    for mol_idx in mols_to_move:
        translation = np.diag(box) * rng.uniform(size=3)
        atom_idxs = np.array(group_idxs[mol_idx])
        moved_coords = randomly_rotate_and_translate(conf[atom_idxs], translation)
        updated_conf = conf.copy()
        updated_conf[atom_idxs] = moved_coords
        test_mol_energies = u_test(updated_conf, box, params)
        ref_mol_energies = u_ref(updated_conf, box, params)
        large_energy_indices = np.argwhere(np.abs(ref_mol_energies) >= threshold)
        comparable_energies = np.delete(np.arange(len(test_mol_energies)), large_energy_indices)
        np.testing.assert_allclose(
            ref_mol_energies[comparable_energies], test_mol_energies[comparable_energies], rtol=rtol, atol=atol
        )
        # Pull out nans, as they are effectively greater than the threshold
        non_nan_idx = np.isfinite(test_mol_energies[large_energy_indices])
        # Large energies are not reliable, so beyond the threshold we simply verify that both the reference and test both exceed the threshold
        assert np.all(np.abs(test_mol_energies[large_energy_indices][non_nan_idx]) >= threshold)
