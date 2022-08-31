# test that end-states are setup correctly in single topology calculations.
import functools
import multiprocessing
import os
from importlib import resources

from timemachine.constants import DEFAULT_FF
from timemachine.ff.handlers import openmm_deserializer

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
    canonicalize_improper_idxs,
    setup_dummy_interactions_from_ff,
)
from timemachine.fe.system import convert_bps_into_system, minimize_scipy, simulate_system
from timemachine.fe.utils import get_mol_name, get_romol_conf
from timemachine.ff import Forcefield
from timemachine.md.builders import build_water_system
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


def assert_bond_idxs_are_canonical(all_idxs):
    for idxs in all_idxs:
        assert idxs[0] < idxs[-1]


def assert_chiral_atom_idxs_are_canonical(all_idxs):
    for _, j, k, l in all_idxs:
        assert (j, k, l) < (l, j, k)
        assert (j, k, l) < (k, l, j)


@pytest.mark.nightly(reason="Takes awhile to run")
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

        print("Checking", get_mol_name(mol_a), "->", get_mol_name(mol_b))
        mcs_threshold = 2.0  # distance threshold, in nanometers
        res = atom_mapping.mcs(mol_a, mol_b, mcs_threshold)
        query = Chem.MolFromSmarts(res.smartsString)
        core_pairs = atom_mapping.get_core_by_mcs(mol_a, mol_b, query, mcs_threshold)
        st = SingleTopologyV3(mol_a, mol_b, core_pairs, ff)
        x0 = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))
        systems = [st.src_system, st.dst_system]

        for system in systems:

            # assert that the idxs are canonicalized.
            assert_bond_idxs_are_canonical(system.bond.get_idxs())
            assert_bond_idxs_are_canonical(system.angle.get_idxs())
            assert_bond_idxs_are_canonical(system.torsion.get_idxs())
            assert_bond_idxs_are_canonical(system.nonbonded.get_idxs())
            assert_bond_idxs_are_canonical(system.chiral_bond.get_idxs())
            assert_chiral_atom_idxs_are_canonical(system.chiral_atom.get_idxs())

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


def test_canonicalize_improper_idxs():

    # these are in the cw rotation set
    improper_idxs = [(5, 0, 1, 3), (5, 1, 3, 0), (5, 3, 0, 1)]

    for idxs in improper_idxs:
        # we should do nothing here.
        assert idxs == canonicalize_improper_idxs(idxs)

    # these are in the ccw rotation set
    #                          1          2          0
    # bad_improper_idxs = [(5,1,0,3), (5,3,1,0), (5,0,3,1)]
    assert canonicalize_improper_idxs((5, 1, 0, 3)) == (5, 1, 3, 0)
    assert canonicalize_improper_idxs((5, 3, 1, 0)) == (5, 3, 0, 1)
    assert canonicalize_improper_idxs((5, 0, 3, 1)) == (5, 0, 1, 3)


def test_combine_masses():

    C_mass = Chem.MolFromSmiles("C").GetAtomWithIdx(0).GetMass()
    Br_mass = Chem.MolFromSmiles("Br").GetAtomWithIdx(0).GetMass()
    F_mass = Chem.MolFromSmiles("F").GetAtomWithIdx(0).GetMass()
    N_mass = Chem.MolFromSmiles("N").GetAtomWithIdx(0).GetMass()

    mol_a = Chem.MolFromSmiles("BrC1=CC=CC=C1")
    mol_b = Chem.MolFromSmiles("C1=CN=CC=C1F")
    core = np.array([[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    st = SingleTopologyV3(mol_a, mol_b, core, ff)

    test_masses = st.combine_masses()
    ref_masses = [Br_mass, C_mass, C_mass, max(C_mass, N_mass), C_mass, C_mass, C_mass, F_mass]
    np.testing.assert_almost_equal(test_masses, ref_masses)


def test_jax_transform_intermediate_potential():
    def setup_arbitary_transformation():
        # NOTE: test system can probably be simplified; we just need
        # any SingleTopologyV3 and conformation
        with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
            suppl = Chem.SDMolSupplier(str(path_to_ligand), removeHs=False)
            mols = {get_mol_name(mol): mol for mol in suppl}

        mol_a = mols["206"]
        mol_b = mols["57"]

        mcs_threshold = 2.0
        res = atom_mapping.mcs(mol_a, mol_b, mcs_threshold)
        query = Chem.MolFromSmarts(res.smartsString)
        core_pairs = atom_mapping.get_core_by_mcs(mol_a, mol_b, query, mcs_threshold)

        ff = Forcefield.load_from_file(DEFAULT_FF)
        st = SingleTopologyV3(mol_a, mol_b, core_pairs, ff)
        conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))
        return st, conf

    st, conf = setup_arbitary_transformation()

    def U(x, lam):
        return st.setup_intermediate_state(lam).get_U_fn()(x)

    _ = jax.jit(U)(conf, 0.1)

    confs = jnp.array([conf for _ in range(10)])
    lambdas = jnp.linspace(0, 1, 10)
    _ = jax.vmap(U)(confs, lambdas)
    _ = jax.jit(jax.vmap(U))(confs, lambdas)


def test_combine_with_host():
    """Verifies that combine_with_host correctly sets up all of the U functions"""
    mol_a = Chem.MolFromSmiles("BrC1=CC=CC=C1")
    mol_b = Chem.MolFromSmiles("C1=CN=CC=C1F")
    core = np.array([[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    solvent_sys, _, _, _ = build_water_system(4.0)

    host_bps, _ = openmm_deserializer.deserialize_system(solvent_sys, cutoff=1.2)

    st = SingleTopologyV3(mol_a, mol_b, core, ff)
    host_system = st.combine_with_host(convert_bps_into_system(host_bps), 0.5)
    # Expect there to be 7 functions, including the chiral bond and chiral atom restraints
    assert len(host_system.get_U_fns()) == 7
