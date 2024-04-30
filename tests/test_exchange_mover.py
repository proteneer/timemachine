from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import pytest

from timemachine.constants import DEFAULT_KT, DEFAULT_WATER_FF
from timemachine.ff.handlers import openmm_deserializer
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.builders import build_water_system
from timemachine.md.exchange import exchange_mover
from timemachine.md.exchange.exchange_mover import delta_r_np
from timemachine.potentials import HarmonicBond

pytestmark = [pytest.mark.nocuda]


@pytest.mark.parametrize("num_lig_atoms", [1, 2, 3, 4, 10])
def test_get_water_idxs(num_lig_atoms):
    system, host_conf, _, _ = build_water_system(3.0, DEFAULT_WATER_FF)
    bps, _ = openmm_deserializer.deserialize_system(system, cutoff=1.2)

    bond_pot = next(bp for bp in bps if isinstance(bp.potential, HarmonicBond)).potential

    all_group_idxs = get_group_indices(get_bond_list(bond_pot), host_conf.shape[0])

    assert exchange_mover.get_water_idxs(all_group_idxs) == all_group_idxs

    additional_ligand_atoms = np.arange(num_lig_atoms) + host_conf.shape[0]
    all_group_idxs.append(additional_ligand_atoms)

    water_idxs = exchange_mover.get_water_idxs(all_group_idxs, ligand_idxs=additional_ligand_atoms)
    assert len(water_idxs) == len(all_group_idxs) - 1


@pytest.mark.parametrize("add_offset", [True, False])
def test_batch_log_weights_incremental(add_offset):
    # test that our trick for computing incremental batched log weights is correct
    np.random.seed(2023)
    W = 111  # num waters
    N = 439  # num atoms
    nb_beta = 1.2
    nb_cutoff = 0.6
    nb_params = np.random.rand(N, 4)
    nb_params[:, 0] -= 0.5
    nb_params[:, 1] *= 0.01
    nb_params[:, -1] = 0
    beta = 1 / DEFAULT_KT

    water_idxs = []
    # Add an offset to the waters to emulate proteins
    offset = 0
    if add_offset:
        offset = N - (W * 3)
    for wi in range(W):
        water_idxs.append([wi * 3 + offset + 0, wi * 3 + offset + 1, wi * 3 + offset + 2])  # has to be contiguous

    bdem = exchange_mover.BDExchangeMove(nb_beta, nb_cutoff, nb_params, water_idxs, beta)

    for _ in range(100):
        conf = np.random.rand(N, 3)
        box = np.eye(3) * 5
        initial_weights = bdem.batch_log_weights(conf, box)
        water_idx = np.random.randint(W)

        old_pos = conf[water_idxs[water_idx]]
        new_pos = old_pos + np.random.rand(1, 3)
        test_log_weights, trial_coords = bdem.batch_log_weights_incremental(
            conf, box, water_idx, new_pos, initial_weights
        )

        new_conf = conf.copy()
        new_conf[water_idxs[water_idx]] = new_pos
        ref_final_weights = bdem.batch_log_weights(new_conf, box)

        np.testing.assert_allclose(trial_coords, new_conf)
        np.testing.assert_allclose(np.array(test_log_weights), np.array(ref_final_weights))


def test_inner_insertion():
    # test that we can insert correctly inside a sphere under PBC
    np.random.seed(2023)
    for _ in range(1000):
        radius = np.random.rand()
        center = np.random.rand(3)
        box = np.eye(3) * np.random.rand()
        new_xyz = exchange_mover.inner_insertion(radius, center, box)
        assert np.linalg.norm(delta_r_np(new_xyz, center, box)) < radius


def test_outer_insertion():
    # test that we can insert correctly outside a sphere but inside the box under PBC
    np.random.seed(2023)
    for _ in range(1000):
        center = np.random.rand(3)
        box = np.eye(3) * np.random.rand()
        # radius has to be smaller than maximum dimension of the box / 2
        # in order for the sphere to be enclosed
        radius = np.random.rand() * (np.amax(box) / 2)
        new_xyz = exchange_mover.outer_insertion(radius, center, box)
        assert np.linalg.norm(delta_r_np(new_xyz, center, box)) >= radius


def test_get_water_groups():
    # test that we can partition waters into their groups correctly
    np.random.seed(2023)
    N = 1000
    W = 231

    water_idxs = []
    for wi in range(W):
        water_idxs.append([wi * 3 + 0, wi * 3 + 1, wi * 3 + 2])  # has to be contiguous

    for _ in range(1000):
        coords = np.random.rand(N, 3)
        box = np.eye(3) * np.random.rand()
        center = np.random.rand(3)
        radius = np.random.rand()

        test_inner_mols, test_outer_mols = exchange_mover.get_water_groups(coords, box, center, water_idxs, radius)
        ref_inner_mols, ref_outer_mols = [], []

        for w_idx, (atom_idxs) in enumerate(water_idxs):
            water_coords = coords[atom_idxs]
            centroid = np.mean(water_coords, axis=0)
            if np.linalg.norm(delta_r_np(centroid, center, box)) < radius:
                ref_inner_mols.append(w_idx)
            else:
                ref_outer_mols.append(w_idx)

        np.testing.assert_equal(test_inner_mols, ref_inner_mols)
        np.testing.assert_equal(test_outer_mols, ref_outer_mols)
