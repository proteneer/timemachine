# test that end-states are setup correctly in single topology calculations.
import functools
import multiprocessing
import os
from importlib import resources

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(multiprocessing.cpu_count())

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from rdkit import Chem

from timemachine.fe import atom_mapping, single_topology_v3
from timemachine.fe.single_topology_v3 import (
    CoreBondChangeWarning,
    MultipleAnchorWarning,
    SingleTopologyV3,
    setup_dummy_interactions_from_ff,
)
from timemachine.fe.system import minimize_scipy, simulate_system
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import Forcefield
from timemachine.potentials.jax_utils import distance


def test_phenol():
    """
    Test that dummy interactions are setup correctly for a phenol. We want to check that bonds and angles
    are present when either a single root anchor is provided, or when a root anchor and a neighbor anchor is provided.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    # set [O,H] as the dummy group
    all_idxs, _ = setup_dummy_interactions_from_ff(
        ff, mol, dummy_group=[6, 12], root_anchor_atom=5, nbr_anchor_atom=None
    )
    bond_idxs, angle_idxs, improper_idxs = all_idxs

    assert set(bond_idxs) == set([(5, 6), (6, 12)])
    assert set(angle_idxs) == set([(5, 6, 12)])
    assert set(improper_idxs) == set()

    # set [O,H] as the dummy group but allow an extra angle
    all_idxs, _ = setup_dummy_interactions_from_ff(ff, mol, dummy_group=[6, 12], root_anchor_atom=5, nbr_anchor_atom=0)
    bond_idxs, angle_idxs, improper_idxs = all_idxs

    assert set(bond_idxs) == set([(5, 6), (6, 12)])
    assert set(angle_idxs) == set([(5, 6, 12), (0, 5, 6)])
    assert set(improper_idxs) == set()

    # set [H] as the dummy group, without neighbor anchor atom
    all_idxs, _ = setup_dummy_interactions_from_ff(ff, mol, dummy_group=[12], root_anchor_atom=6, nbr_anchor_atom=None)
    bond_idxs, angle_idxs, improper_idxs = all_idxs

    assert set(bond_idxs) == set([(6, 12)])
    assert set(angle_idxs) == set()
    assert set(improper_idxs) == set()

    # set [H] as the dummy group, with neighbor anchor atom
    all_idxs, _ = setup_dummy_interactions_from_ff(ff, mol, dummy_group=[12], root_anchor_atom=6, nbr_anchor_atom=5)
    bond_idxs, angle_idxs, improper_idxs = all_idxs

    assert set(bond_idxs) == set([(6, 12)])
    assert set(angle_idxs) == set([(5, 6, 12)])
    assert set(improper_idxs) == set()

    with pytest.raises(single_topology_v3.MissingAngleError):
        all_idxs, _ = setup_dummy_interactions_from_ff(ff, mol, dummy_group=[12], root_anchor_atom=6, nbr_anchor_atom=4)


def test_find_dummy_groups_and_anchors():
    """
    Test that we can find the anchors and dummy groups when there's a single core anchor atom. When core bond
    is broken, we should disable one of the angle atoms.
    """
    mol_a = Chem.MolFromSmiles("OCCC")
    mol_b = Chem.MolFromSmiles("CCCF")
    core_pairs = np.array([[1, 2], [2, 1], [3, 0]])

    dgs, jks = single_topology_v3.find_dummy_groups_and_anchors(mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1])
    assert dgs == [{3}]
    assert jks == [(2, 1)]

    # angle should swap
    core_pairs = np.array([[1, 2], [2, 0], [3, 1]])

    with pytest.warns(CoreBondChangeWarning):
        dgs, jks = single_topology_v3.find_dummy_groups_and_anchors(mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1])
        assert dgs == [{3}]
        assert jks == [(2, None)]


def test_find_dummy_groups_and_anchors_multiple_angles():
    """
    Test that when multiple angle groups are possible we can find one deterministically
    """
    mol_a = Chem.MolFromSmiles("CCC")
    mol_b = Chem.MolFromSmiles("CC(C)C")

    core_pairs = np.array([[0, 2], [1, 1], [2, 3]])
    dgs, jks = single_topology_v3.find_dummy_groups_and_anchors(mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1])
    assert dgs == [{0}]
    assert jks == [(1, 2)] or jks == [(1, 3)]

    dgs_zero, jks_zero = single_topology_v3.find_dummy_groups_and_anchors(
        mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1]
    )

    # this code should be invariant to different random seeds and different ordering of core pairs
    for idx in range(100):
        np.random.seed(idx)
        core_pairs_shuffle = np.random.permutation(core_pairs)
        dgs, jks = single_topology_v3.find_dummy_groups_and_anchors(
            mol_a, mol_b, core_pairs_shuffle[:, 0], core_pairs_shuffle[:, 1]
        )
        assert dgs == dgs_zero
        assert jks == jks_zero


def testing_find_dummy_groups_and_multiple_anchors():
    """
    Test that we can find anchors and dummy groups with multiple anchors, we expect to find only a single
    root anchor and neighbor core atom pair.
    """
    mol_a = Chem.MolFromSmiles("OCC")
    mol_b = Chem.MolFromSmiles("O1CC1")
    core_pairs = np.array([[1, 1], [2, 2]])

    with pytest.warns(MultipleAnchorWarning):
        dgs, jks = single_topology_v3.find_dummy_groups_and_anchors(mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1])
        assert dgs == [{0}]
        assert jks == [(1, 2)] or jks == [(2, 1)]

    # test determinism, should be robust against seeds
    dgs_zero, jks_zero = single_topology_v3.find_dummy_groups_and_anchors(
        mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1]
    )
    for idx in range(100):
        np.random.seed(idx)
        core_pairs_shuffle = np.random.permutation(core_pairs)
        dgs, jks = single_topology_v3.find_dummy_groups_and_anchors(
            mol_a, mol_b, core_pairs_shuffle[:, 0], core_pairs_shuffle[:, 1]
        )
        assert dgs == dgs_zero
        assert jks == jks_zero

    mol_a = Chem.MolFromSmiles("C(C)(C)C")
    mol_b = Chem.MolFromSmiles("O1CCCC1")

    core_a = [0, 1, 2, 3]
    core_b = [2, 1, 4, 3]

    with pytest.warns(MultipleAnchorWarning):
        dgs, jks = single_topology_v3.find_dummy_groups_and_anchors(mol_a, mol_b, core_a, core_b)
        assert dgs == [{0}]
        assert jks == [(1, 2)]


def test_hif2a_end_state_stability(num_pairs_to_setup=25, num_pairs_to_simulate=5):
    """
    Pick some random pairs from the hif2a set and ensure that they're numerically stable at the
    end-states under a distance based atom-mapping protocol. For a subset of them, we will also run
    simulations.
    """

    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        suppl = Chem.SDMolSupplier(str(path_to_ligand), removeHs=False)
        mols = [m for m in suppl]

    pairs = [(mol_a, mol_b) for mol_a in mols for mol_b in mols]

    np.random.seed(2023)
    np.random.shuffle(pairs)
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    compute_distance_matrix = functools.partial(distance, box=None)

    def get_max_distance(x0):
        dij = compute_distance_matrix(x0)
        return jnp.amax(dij)

    batch_distance_check = jax.vmap(get_max_distance)

    # this has been tested for up to 50 random pairs
    for pair_idx, (mol_a, mol_b) in enumerate(pairs[:num_pairs_to_setup]):

        print("Checking", mol_a.GetProp("_Name"), "->", mol_b.GetProp("_Name"))
        mcs_threshold = 0.75  # distance threshold, in nanometers
        res = atom_mapping.mcs_map(mol_a, mol_b, mcs_threshold)
        query = Chem.MolFromSmarts(res.smartsString)
        core_pairs = atom_mapping.get_core_by_mcs(mol_a, mol_b, query, mcs_threshold)
        st = SingleTopologyV3(mol_a, mol_b, core_pairs, ff)
        x0 = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))
        systems = [
            st.setup_end_state_src(),
            st.setup_end_state_dst(),
        ]

        for system in systems:
            U_fn = jax.jit(system.get_U_fn())
            assert np.isfinite(U_fn(x0))
            x_min = minimize_scipy(U_fn, x0)
            assert np.all(np.isfinite(x_min))
            distance_cutoff = 2.5  # in nanometers
            assert get_max_distance(x_min) < distance_cutoff

            # test running simulations on the first 5 pairs
            if pair_idx < num_pairs_to_simulate:
                batch_U_fn = jax.vmap(U_fn)
                frames = simulate_system(system.get_U_fn(), x0, num_samples=1000)
                nrgs = batch_U_fn(frames)
                assert np.all(np.isfinite(nrgs))
                assert np.all(np.isfinite(frames))
                assert np.all(batch_distance_check(frames) < distance_cutoff)
