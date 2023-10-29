from importlib import resources

import numpy as np
import pytest

from timemachine.constants import DEFAULT_PRESSURE, DEFAULT_TEMP
from timemachine.fe.model_utils import apply_hmr
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops
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


@pytest.mark.parametrize("precision,atol,rtol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 3e-4)])
def test_nonbonded_mol_energy_matches_exchange_mover_batch_U_in_complex(precision, atol, rtol):
    """Test that computing the per water energies of a system with a complex is equivalent."""
    ff = Forcefield.load_default()
    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_ligand:
        complex_system, conf, box, _, _ = builders.build_protein_system(str(path_to_ligand), ff.protein_ff, ff.water_ff)
    bps, masses = openmm_deserializer.deserialize_system(complex_system, cutoff=1.2)
    nb = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    bond_list = get_bond_list(bond_pot)
    all_group_idxs = get_group_indices(bond_list, conf.shape[0])

    # Equilibrate the system a bit before hand, which reduces clashes in the system which results greater differences
    # between the reference and test case.
    seed = 2023
    dt = 1.5e-3
    temperature = DEFAULT_TEMP
    pressure = DEFAULT_PRESSURE

    masses = apply_hmr(masses, bond_list)
    intg = LangevinIntegrator(temperature, dt, 1.0, np.array(masses), seed).impl()

    bound_impls = []

    for potential in bps:
        bound_impls.append(potential.to_gpu(precision=np.float32).bound_impl)  # get the bound implementation

    barostat_interval = 5
    baro = MonteCarloBarostat(
        conf.shape[0],
        pressure,
        temperature,
        all_group_idxs,
        barostat_interval,
        seed,
    )
    baro_impl = baro.impl(bound_impls)

    ctxt = custom_ops.Context(
        conf,
        np.zeros_like(conf),
        box,
        intg,
        bound_impls,
        barostat=baro_impl,
    )
    ctxt.multiple_steps(1000)
    conf = ctxt.get_x_t()
    box = ctxt.get_box()

    # only act on waters
    water_groups = [group for group in all_group_idxs if len(group) == 3]

    N = conf.shape[0]

    params = nb.params

    beta = nb.potential.beta
    cutoff = nb.potential.cutoff
    klass = custom_ops.NonbondedMolEnergyPotential_f32
    if precision == np.float64:
        klass = custom_ops.NonbondedMolEnergyPotential_f64

    mover = BDExchangeMove(beta, cutoff, params, water_groups, DEFAULT_TEMP)

    mol_by_mol_pot = klass(N, water_groups, beta, cutoff)

    def u_ref(x, box, params):
        return mover.batch_U_fn(x, box, mover.all_a_idxs, mover.all_b_idxs)

    def u_test(x, box, params):
        mol_energies = mol_by_mol_pot.execute(x, params, box)
        assert mol_energies.shape == (len(water_groups),)

        # Make sure running again gives bitwise identical results
        comp_mol_energies = mol_by_mol_pot.execute(x, params, box)
        np.testing.assert_array_equal(mol_energies, comp_mol_energies)
        return mol_energies

    np.testing.assert_allclose(u_test(conf, box, params), u_ref(conf, box, params), rtol=rtol, atol=atol)
