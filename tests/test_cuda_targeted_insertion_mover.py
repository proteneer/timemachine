import numpy as np
import pytest

from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import custom_ops
from timemachine.md import builders
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.exchange.exchange_mover import delta_r_np, get_water_groups
from timemachine.potentials import HarmonicBond


@pytest.mark.parametrize("seed", [2023, 2024])
@pytest.mark.parametrize("radius", [0.1, 0.5, 1.2, 2.0])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_inner_and_outer_water_groups(seed, radius, precision):
    rng = np.random.default_rng(seed)
    ff = Forcefield.load_default()
    system, coords, box, _ = builders.build_water_system(4.0, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), coords.shape[0])

    center_group_idx = rng.choice(np.arange(len(all_group_idxs)))

    center_group = all_group_idxs.pop(center_group_idx)

    group_idxs = np.delete(np.array(all_group_idxs).reshape(-1), center_group)
    group_idxs = group_idxs.reshape(len(all_group_idxs) - 1, 3)

    center = np.mean(coords[center_group], axis=0)

    ref_inner, ref_outer = get_water_groups(coords, box, center, group_idxs, radius)

    func = custom_ops.inner_and_outer_mols_f32
    if precision == np.float64:
        func = custom_ops.inner_and_outer_mols_f64

    inner_mol_idxs, outer_mol_idxs = func(center_group, coords, box, group_idxs, radius)

    assert len(inner_mol_idxs) + len(outer_mol_idxs) == len(group_idxs)
    np.testing.assert_equal(list(sorted(ref_inner)), list(sorted(inner_mol_idxs)))
    np.testing.assert_equal(list(sorted(ref_outer)), list(sorted(outer_mol_idxs)))


@pytest.mark.memcheck
@pytest.mark.parametrize("seed", [2023])
@pytest.mark.parametrize("n_translations", [1, 1000])
@pytest.mark.parametrize("radius", [1.0, 2.0])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_translations_within_sphere(seed, n_translations, radius, precision):
    rng = np.random.default_rng(seed)
    ff = Forcefield.load_default()
    system, coords, box, _ = builders.build_water_system(4.0, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), coords.shape[0])

    center_group_idx = rng.choice(np.arange(len(all_group_idxs)))

    center_group = all_group_idxs.pop(center_group_idx)

    center = np.mean(coords[center_group], axis=0)

    func = custom_ops.translation_within_sphere_f32
    if precision == np.float64:
        func = custom_ops.translation_within_sphere_f64

    translations_a = func(n_translations, center, radius, seed)
    translations_b = func(n_translations, center, radius, seed)
    # Bitwise deterministic with a provided seed
    np.testing.assert_array_equal(translations_a, translations_b)

    last_translation = None
    for translation in translations_a:
        assert np.linalg.norm(delta_r_np(translation, center, box)) < radius
        if last_translation is not None:
            assert not np.all(last_translation == translation)


@pytest.mark.memcheck
@pytest.mark.parametrize("seed", [2023])
@pytest.mark.parametrize("n_translations", [1, 32])
@pytest.mark.parametrize("radius", [1.0, 2.0])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_translations_outside_sphere(seed, n_translations, radius, precision):
    rng = np.random.default_rng(seed)
    ff = Forcefield.load_default()
    system, coords, box, _ = builders.build_water_system(4.0, ff.water_ff)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), coords.shape[0])

    center_group_idx = rng.choice(np.arange(len(all_group_idxs)))

    center_group = all_group_idxs.pop(center_group_idx)

    center = np.mean(coords[center_group], axis=0)

    func = custom_ops.translation_outside_sphere_f32
    if precision == np.float64:
        func = custom_ops.translation_outside_sphere_f64

    translations_a = func(n_translations, center, box, radius, seed)
    translations_b = func(n_translations, center, box, radius, seed)
    # Bitwise deterministic with a provided seed
    np.testing.assert_array_equal(translations_a, translations_b)

    last_translation = None
    for i, translation in enumerate(translations_a):
        assert np.linalg.norm(delta_r_np(translation, center, box)) >= radius, str(i)
        if last_translation is not None:
            assert not np.all(last_translation == translation)
