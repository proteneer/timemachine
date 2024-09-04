import functools
import time
from importlib import resources

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest
from common import check_split_ixns, load_split_forcefields
from hypothesis import assume, event, given, seed
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine import potentials
from timemachine.constants import (
    DEFAULT_ATOM_MAPPING_KWARGS,
    DEFAULT_CHIRAL_ATOM_RESTRAINT_K,
    DEFAULT_CHIRAL_BOND_RESTRAINT_K,
)
from timemachine.fe import atom_mapping, single_topology
from timemachine.fe.dummy import MultipleAnchorWarning
from timemachine.fe.free_energy import HostConfig
from timemachine.fe.interpolate import (
    align_harmonic_bond_idxs_and_params,
    align_nonbonded_idxs_and_params,
    linear_interpolation,
    log_linear_interpolation,
)
from timemachine.fe.single_topology import (
    ChargePertubationError,
    CoreBondChangeWarning,
    DummyGroupAssignmentError,
    SingleTopology,
    canonicalize_bonds,
    canonicalize_chiral_atom_idxs,
    canonicalize_improper_idxs,
    cyclic_difference,
    handle_ring_opening_closing,
    interpolate_harmonic_bond_params,
    interpolate_harmonic_force_constant,
    interpolate_w_coord,
    setup_dummy_interactions_from_ff,
    verify_chiral_validity_of_core,
)
from timemachine.fe.system import convert_bps_into_system, minimize_scipy, simulate_system
from timemachine.fe.utils import get_mol_name, get_romol_conf, read_sdf, set_mol_name
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.md import minimizer
from timemachine.md.builders import build_protein_system, build_water_system
from timemachine.potentials.jax_utils import pairwise_distances

setup_chiral_dummy_interactions_from_ff = functools.partial(
    setup_dummy_interactions_from_ff,
    chiral_atom_k=DEFAULT_CHIRAL_ATOM_RESTRAINT_K,
    chiral_bond_k=DEFAULT_CHIRAL_BOND_RESTRAINT_K,
)


@pytest.mark.nocuda
def test_setup_chiral_dummy_atoms():
    """
    Test that we setup the correct geometries for each of the 8 types specified in single topology
    """
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    mol = Chem.MolFromMolBlock(
        """
  Mrv2311 02232401393D

  5  4  0  0  0  0            999 V2000
    1.8515    0.0946    2.1705 F   0  0  0  0  0  0  0  0  0  0  0  0
    1.0043    0.5689    1.2025 C   0  0  2  0  0  0  0  0  0  0  0  0
   -0.6276    0.0025    1.5238 Cl  0  0  0  0  0  0  0  0  0  0  0  0
    1.5780   -0.0702   -0.5225 Br  0  0  0  0  0  0  0  0  0  0  0  0
    1.0321    2.6887    1.2159 I   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  2  4  1  0  0  0  0
  2  5  1  0  0  0  0
M  END
$$$$"""
    )

    # First 3 tests, center is core
    dg_0 = [0, 2, 3]
    core_0 = [1]
    idxs, params = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dg_0, root_anchor_atom=1, nbr_anchor_atom=None, core_atoms=core_0
    )
    chiral_atom_idxs = idxs[-1]
    chiral_atom_params = params[-1]
    np.testing.assert_array_equal(chiral_atom_idxs, [[1, 2, 0, 3]])
    np.testing.assert_array_equal(chiral_atom_params, [DEFAULT_CHIRAL_ATOM_RESTRAINT_K])

    dg_1 = [0, 3]
    core_1 = [1, 2]
    idxs, params = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dg_1, root_anchor_atom=1, nbr_anchor_atom=None, core_atoms=core_1
    )
    chiral_atom_idxs = idxs[-1]
    chiral_atom_params = params[-1]
    np.testing.assert_array_equal(chiral_atom_idxs, [[1, 2, 0, 3]])
    np.testing.assert_array_equal(chiral_atom_params, [DEFAULT_CHIRAL_ATOM_RESTRAINT_K])

    dg_2 = [0]
    core_2 = [1, 2, 3]
    idxs, params = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dg_2, root_anchor_atom=1, nbr_anchor_atom=None, core_atoms=core_2
    )
    chiral_atom_idxs = idxs[-1]
    chiral_atom_params = params[-1]
    np.testing.assert_array_equal(chiral_atom_idxs, [[1, 2, 0, 3]])
    np.testing.assert_array_equal(chiral_atom_params, [DEFAULT_CHIRAL_ATOM_RESTRAINT_K])

    # Next 3 tests, center is not core
    dg_3 = [1, 2, 3]
    core_3 = [0]
    idxs, params = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dg_3, root_anchor_atom=0, nbr_anchor_atom=None, core_atoms=core_3
    )
    chiral_atom_idxs = idxs[-1]
    chiral_atom_params = params[-1]
    np.testing.assert_array_equal(chiral_atom_idxs, [[1, 2, 0, 3]])
    np.testing.assert_array_equal(chiral_atom_params, [DEFAULT_CHIRAL_ATOM_RESTRAINT_K])

    dg_4 = [1, 2]
    core_4 = [0, 3]
    idxs, params = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dg_4, root_anchor_atom=0, nbr_anchor_atom=None, core_atoms=core_4
    )
    chiral_atom_idxs = idxs[-1]
    chiral_atom_params = params[-1]
    np.testing.assert_array_equal(chiral_atom_idxs, [[1, 2, 0, 3]])
    np.testing.assert_array_equal(chiral_atom_params, [DEFAULT_CHIRAL_ATOM_RESTRAINT_K])

    dg_5 = [0, 1, 2, 3]
    core_5 = [4]
    idxs, params = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dg_5, root_anchor_atom=4, nbr_anchor_atom=None, core_atoms=core_5
    )
    chiral_atom_idxs = idxs[-1]
    chiral_atom_params = params[-1]
    np.testing.assert_array_equal(chiral_atom_idxs, [[1, 2, 0, 3], [1, 0, 2, 4], [1, 3, 0, 4], [1, 2, 3, 4]])
    np.testing.assert_array_equal(chiral_atom_params, [DEFAULT_CHIRAL_ATOM_RESTRAINT_K] * 4)

    # The next two should return empty
    dg_6 = [1]
    core_6 = [0, 2, 3]
    idxs, params = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dg_6, root_anchor_atom=0, nbr_anchor_atom=None, core_atoms=core_6
    )
    chiral_atom_idxs = idxs[-1]
    chiral_atom_params = params[-1]
    assert len(chiral_atom_idxs) == 0
    assert len(chiral_atom_params) == 0

    dg_7 = []
    core_7 = [0, 1, 2, 3]
    idxs, params = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dg_7, root_anchor_atom=0, nbr_anchor_atom=None, core_atoms=core_7
    )
    chiral_atom_idxs = idxs[-1]
    chiral_atom_params = params[-1]
    assert len(chiral_atom_idxs) == 0
    assert len(chiral_atom_params) == 0


def assert_bond_sets_equal(bonds_a, bonds_b):
    def f(bonds):
        return {tuple(idxs) for idxs in bonds}

    return f(bonds_a) == f(bonds_b)


@pytest.mark.nocuda
def test_phenol():
    """
    Test that dummy interactions are setup correctly for a phenol. We want to check that bonds and angles
    are present when either a single root anchor is provided, or when a root anchor and a neighbor anchor is provided.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    AllChem.EmbedMolecule(mol, randomSeed=2022)

    all_atoms_set = set([a for a in range(mol.GetNumAtoms())])

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    dg_0 = [6, 12]
    core_0 = list(all_atoms_set.difference(dg_0))

    # set [O,H] as the dummy group
    all_idxs, _ = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dummy_group=dg_0, root_anchor_atom=5, nbr_anchor_atom=None, core_atoms=core_0
    )
    bond_idxs, angle_idxs, improper_idxs, chiral_atom_idxs = all_idxs

    assert_bond_sets_equal(bond_idxs, [(5, 6), (6, 12)])
    assert_bond_sets_equal(angle_idxs, [(5, 6, 12)])
    assert len(improper_idxs) == 0
    assert len(chiral_atom_idxs) == 0

    # set [O,H] as the dummy group but allow an extra angle
    all_idxs, _ = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dummy_group=dg_0, root_anchor_atom=5, nbr_anchor_atom=0, core_atoms=core_0
    )
    bond_idxs, angle_idxs, improper_idxs, chiral_atom_idxs = all_idxs

    assert_bond_sets_equal(bond_idxs, [(5, 6), (6, 12)])
    assert_bond_sets_equal(angle_idxs, [(5, 6, 12), (0, 5, 6)])
    assert len(improper_idxs) == 0
    assert len(chiral_atom_idxs) == 0

    dg_1 = [12]
    core_1 = list(all_atoms_set.difference(dg_1))

    # set [H] as the dummy group, without neighbor anchor atom
    all_idxs, _ = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dummy_group=dg_1, root_anchor_atom=6, nbr_anchor_atom=None, core_atoms=core_1
    )
    bond_idxs, angle_idxs, improper_idxs, chiral_atom_idxs = all_idxs

    assert_bond_sets_equal(bond_idxs, [(6, 12)])
    assert len(angle_idxs) == 0
    assert len(improper_idxs) == 0
    assert len(chiral_atom_idxs) == 0

    # set [H] as the dummy group, with neighbor anchor atom
    all_idxs, _ = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dummy_group=dg_1, root_anchor_atom=6, nbr_anchor_atom=5, core_atoms=core_1
    )
    bond_idxs, angle_idxs, improper_idxs, chiral_atom_idxs = all_idxs

    assert_bond_sets_equal(bond_idxs, [(6, 12)])
    assert_bond_sets_equal(angle_idxs, [(5, 6, 12)])
    assert len(improper_idxs) == 0
    assert len(chiral_atom_idxs) == 0

    with pytest.raises(single_topology.MissingAngleError):
        all_idxs, _ = setup_chiral_dummy_interactions_from_ff(
            ff, mol, dummy_group=dg_1, root_anchor_atom=6, nbr_anchor_atom=4, core_atoms=core_1
        )


@pytest.mark.nocuda
def test_methyl_chiral_atom_idxs():
    """
    Check that we're leaving the chiral restraints on correctly for a methyl, when only a single hydrogen is a core atom.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("C"))
    AllChem.EmbedMolecule(mol, randomSeed=2022)

    dg = [1, 2, 3, 4]
    all_atoms_set = set([a for a in range(mol.GetNumAtoms())])
    core_atoms = list(all_atoms_set.difference(dg))

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    # set [O,H] as the dummy group
    all_idxs, _ = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dummy_group=dg, root_anchor_atom=0, nbr_anchor_atom=None, core_atoms=core_atoms
    )
    _, _, _, chiral_atom_idxs = all_idxs

    expected_chiral_atom_idxs = [
        [
            (0, 1, 3, 4),
            (0, 3, 2, 4),
            (0, 2, 1, 4),
            (0, 1, 2, 3),
        ]
    ]

    assert_bond_sets_equal(chiral_atom_idxs, expected_chiral_atom_idxs)


@pytest.mark.nocuda
def test_find_dummy_groups_and_anchors():
    """
    Test that we can find the anchors and dummy groups when there's a single core anchor atom. When core bond
    is broken, we should disable one of the angle atoms.
    """
    mol_a = Chem.MolFromSmiles("OCCC")
    mol_b = Chem.MolFromSmiles("CCCF")

    AllChem.EmbedMolecule(mol_a, randomSeed=2022)
    AllChem.EmbedMolecule(mol_b, randomSeed=2022)

    core_pairs = np.array([[1, 2], [2, 1], [3, 0]])

    dgs = single_topology.find_dummy_groups_and_anchors(mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1])
    assert dgs == {2: (1, {3})}

    # angle should swap
    core_pairs = np.array([[1, 2], [2, 0], [3, 1]])

    with pytest.warns(CoreBondChangeWarning):
        dgs = single_topology.find_dummy_groups_and_anchors(mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1])
        assert dgs == {2: (None, {3})}


@pytest.mark.nocuda
def test_find_dummy_groups_and_anchors_multiple_angles():
    """
    Test that when multiple angle groups are possible we can find one deterministically
    """
    mol_a = Chem.MolFromSmiles("CCC")
    mol_b = Chem.MolFromSmiles("CC(C)C")

    AllChem.EmbedMolecule(mol_a, randomSeed=2022)
    AllChem.EmbedMolecule(mol_b, randomSeed=2022)

    core_pairs = np.array([[0, 2], [1, 1], [2, 3]])
    dgs = single_topology.find_dummy_groups_and_anchors(mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1])
    assert dgs == {1: (2, {0})} or dgs == {1: (3, {0})}

    dgs_zero = single_topology.find_dummy_groups_and_anchors(mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1])

    # this code should be invariant to different random seeds and different ordering of core pairs
    for idx in range(100):
        np.random.seed(idx)
        core_pairs_shuffle = np.random.permutation(core_pairs)
        dgs = single_topology.find_dummy_groups_and_anchors(
            mol_a, mol_b, core_pairs_shuffle[:, 0], core_pairs_shuffle[:, 1]
        )
        assert dgs == dgs_zero


@pytest.mark.nocuda
def test_find_dummy_groups_and_multiple_anchors():
    """
    Test that we can find anchors and dummy groups with multiple anchors, we expect to find only a single
    root anchor and neighbor core atom pair.
    """
    mol_a = Chem.MolFromSmiles("OCC")
    mol_b = Chem.MolFromSmiles("O1CC1")

    AllChem.EmbedMolecule(mol_a, randomSeed=2022)
    AllChem.EmbedMolecule(mol_b, randomSeed=2022)

    core_pairs = np.array([[1, 1], [2, 2]])

    with pytest.warns(MultipleAnchorWarning):
        dgs = single_topology.find_dummy_groups_and_anchors(mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1])
        assert dgs == {1: (2, {0})} or dgs == {2: (1, {0})}

    # test determinism, should be robust against seeds
    dgs_zero = single_topology.find_dummy_groups_and_anchors(mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1])
    for idx in range(100):
        np.random.seed(idx)
        core_pairs_shuffle = np.random.permutation(core_pairs)
        dgs = single_topology.find_dummy_groups_and_anchors(
            mol_a, mol_b, core_pairs_shuffle[:, 0], core_pairs_shuffle[:, 1]
        )
        assert dgs == dgs_zero

    mol_a = Chem.MolFromSmiles("C(C)(C)C")
    mol_b = Chem.MolFromSmiles("O1CCCC1")

    AllChem.EmbedMolecule(mol_a, randomSeed=2022)
    AllChem.EmbedMolecule(mol_b, randomSeed=2022)

    core_a = [0, 1, 2, 3]
    core_b = [2, 1, 4, 3]

    with pytest.warns(MultipleAnchorWarning):
        dgs = single_topology.find_dummy_groups_and_anchors(mol_a, mol_b, core_a, core_b)
        assert dgs == {1: (2, {0})}


@pytest.mark.nocuda
def test_ethane_cyclobutadiene():
    """Test case where a naive heuristic for identifying dummy groups results in disconnected components"""

    mol_a = Chem.AddHs(Chem.MolFromSmiles("CC"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1ccc1"))

    AllChem.EmbedMolecule(mol_a, randomSeed=2022)
    AllChem.EmbedMolecule(mol_b, randomSeed=2022)

    core = np.array([[2, 0], [4, 2], [0, 3], [3, 7]])
    ff = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, ff)

    g = nx.Graph()
    g.add_nodes_from(range(st.get_num_atoms()))
    g.add_edges_from(st.src_system.bond.potential.idxs)

    # bond graph should be connected (i.e. no floating bits)
    assert len(list(nx.connected_components(g))) == 1


@pytest.mark.nocuda
def test_charge_perturbation_is_invalid():
    mol_a = Chem.AddHs(Chem.MolFromSmiles("Cc1cc[nH]c1"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("C[n+]1cc[nH]c1"))

    AllChem.EmbedMolecule(mol_a, randomSeed=2022)
    AllChem.EmbedMolecule(mol_b, randomSeed=2022)

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    core = np.zeros((mol_a.GetNumAtoms(), 2), dtype=np.int32)
    core[:, 0] = np.arange(core.shape[0])
    core[:, 1] = core[:, 0]

    with pytest.raises(ChargePertubationError) as e:
        SingleTopology(mol_a, mol_b, core, ff)
    assert str(e.value) == "mol a and mol b don't have the same charge: a: 0 b: 1"


def bond_idxs_are_canonical(all_idxs):
    return np.all(all_idxs[:, 0] < all_idxs[:, -1])


def chiral_atom_idxs_are_canonical(all_idxs):
    return np.all((all_idxs[:, 1] < all_idxs[:, 2]) & (all_idxs[:, 1] < all_idxs[:, 3]))


@pytest.mark.nogpu
@pytest.mark.nightly(reason="Takes awhile to run")
def test_hif2a_end_state_stability(num_pairs_to_setup=25, num_pairs_to_simulate=5):
    """
    Pick some random pairs from the hif2a set and ensure that they're numerically stable at the
    end-states under a distance based atom-mapping protocol. For a subset of them, we will also run
    simulations.
    """

    seed = 2024

    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols = read_sdf(path_to_ligand)

    pairs = [(mol_a, mol_b) for mol_a in mols for mol_b in mols]

    np.random.seed(seed)
    np.random.shuffle(pairs)
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    compute_distance_matrix = functools.partial(pairwise_distances, box=None)

    def get_max_distance(x0):
        dij = compute_distance_matrix(x0)
        return jnp.amax(dij)

    batch_distance_check = jax.vmap(get_max_distance)

    # this has been tested for up to 50 random pairs
    for pair_idx, (mol_a, mol_b) in enumerate(pairs[:num_pairs_to_setup]):
        print("Checking", get_mol_name(mol_a), "->", get_mol_name(mol_b))
        core = _get_core_by_mcs(mol_a, mol_b)
        st = SingleTopology(mol_a, mol_b, core, ff)
        x0 = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))
        systems = [st.src_system, st.dst_system]

        for system in systems:
            # assert that the idxs are canonicalized.
            assert bond_idxs_are_canonical(system.bond.potential.idxs)
            assert bond_idxs_are_canonical(system.angle.potential.idxs)
            assert bond_idxs_are_canonical(system.torsion.potential.idxs)
            assert bond_idxs_are_canonical(system.nonbonded.potential.idxs)
            assert bond_idxs_are_canonical(system.chiral_bond.potential.idxs)
            assert chiral_atom_idxs_are_canonical(system.chiral_atom.potential.idxs)
            U_fn = jax.jit(system.get_U_fn())
            assert np.isfinite(U_fn(x0))
            x_min = minimize_scipy(U_fn, x0, seed=seed)
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


atom_idxs = st.integers(0, 100)


@st.composite
def bond_or_angle_idx_arrays(draw):
    n_idxs = draw(st.one_of(st.just(2), st.just(3)))
    idxs = st.lists(atom_idxs, min_size=n_idxs, max_size=n_idxs, unique=True).map(tuple)
    idx_arrays = st.lists(idxs, min_size=0, max_size=100, unique=True).map(
        lambda ixns: np.array(ixns).reshape(-1, n_idxs)
    )
    return draw(idx_arrays)


@given(bond_or_angle_idx_arrays())
@seed(2024)
def test_canonicalize_bonds(bonds):
    canonicalized_bonds = canonicalize_bonds(bonds)
    event("canonical" if bond_idxs_are_canonical(bonds) else "not canonical")
    assert all(set(canon_idxs) == set(idxs) for canon_idxs, idxs in zip(canonicalized_bonds, bonds))
    assert bond_idxs_are_canonical(canonicalized_bonds)


chiral_atom_idxs = st.lists(atom_idxs, min_size=4, max_size=4, unique=True).map(lambda x: tuple(x))
chiral_atom_idx_arrays = st.lists(chiral_atom_idxs, min_size=0, max_size=100, unique=True).map(
    lambda idxs: np.array(idxs).reshape(-1, 4)
)


@given(chiral_atom_idx_arrays)
@seed(2024)
def test_canonicalize_chiral_atom_idxs(chiral_atom_idxs):
    canonicalized_idxs = canonicalize_chiral_atom_idxs(chiral_atom_idxs)
    event("canonical" if chiral_atom_idxs_are_canonical(chiral_atom_idxs) else "not canonical")
    assert all(
        tuple(canon_idxs) in {tuple(idxs[p]) for p in [[0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]}
        for canon_idxs, idxs in zip(canonicalized_idxs, chiral_atom_idxs)
    )
    assert chiral_atom_idxs_are_canonical(canonicalized_idxs)


@pytest.mark.nocuda
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


@pytest.mark.nocuda
def test_combine_masses():
    C_mass = Chem.MolFromSmiles("C").GetAtomWithIdx(0).GetMass()
    Br_mass = Chem.MolFromSmiles("Br").GetAtomWithIdx(0).GetMass()
    F_mass = Chem.MolFromSmiles("F").GetAtomWithIdx(0).GetMass()
    N_mass = Chem.MolFromSmiles("N").GetAtomWithIdx(0).GetMass()

    mol_a = Chem.MolFromSmiles("BrC1=CC=CC=C1")
    mol_b = Chem.MolFromSmiles("C1=CN=CC=C1F")

    AllChem.EmbedMolecule(mol_a, randomSeed=2022)
    AllChem.EmbedMolecule(mol_b, randomSeed=2022)

    core = np.array([[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    st = SingleTopology(mol_a, mol_b, core, ff)

    test_masses = st.combine_masses()
    ref_masses = [Br_mass, C_mass, C_mass, max(C_mass, N_mass), C_mass, C_mass, C_mass, F_mass]
    np.testing.assert_almost_equal(test_masses, ref_masses)


@pytest.mark.nocuda
def test_combine_masses_hmr():
    C_mass = Chem.MolFromSmiles("C").GetAtomWithIdx(0).GetMass()
    Cl_mass = Chem.MolFromSmiles("Cl").GetAtomWithIdx(0).GetMass()
    Br_mass = Chem.MolFromSmiles("Br").GetAtomWithIdx(0).GetMass()
    F_mass = Chem.MolFromSmiles("F").GetAtomWithIdx(0).GetMass()

    mol_a = ligand_from_smiles("[H]C([H])([H])[H]")
    mol_b = ligand_from_smiles("[H]C(F)(Cl)Br")
    H_mass = mol_a.GetAtomWithIdx(1).GetMass()

    AllChem.EmbedMolecule(mol_a, randomSeed=2023)
    AllChem.EmbedMolecule(mol_b, randomSeed=2023)

    # only C mapped
    core = np.array([[0, 0]])

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, ff)

    # No HMR
    test_masses = st.combine_masses()
    ref_masses = [C_mass, H_mass, H_mass, H_mass, H_mass, F_mass, Cl_mass, Br_mass, H_mass]
    np.testing.assert_almost_equal(test_masses, ref_masses)

    # HMR
    test_masses = st.combine_masses(use_hmr=True)
    scale = 2 * H_mass
    ref_masses = [
        max(C_mass - 4 * scale, C_mass - scale),
        H_mass + scale,
        H_mass + scale,
        H_mass + scale,
        H_mass + scale,
        F_mass,
        Cl_mass,
        Br_mass,
        H_mass + scale,
    ]
    np.testing.assert_almost_equal(test_masses, ref_masses)

    # only C-H/C-F mapped
    core = np.array([[0, 0], [1, 1]])

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, ff)

    # No HMR
    test_masses = st.combine_masses()
    ref_masses = [C_mass, F_mass, H_mass, H_mass, H_mass, Cl_mass, Br_mass, H_mass]
    np.testing.assert_almost_equal(test_masses, ref_masses)

    # HMR
    test_masses = st.combine_masses(use_hmr=True)
    ref_masses = [
        max(C_mass - 4 * scale, C_mass - scale),
        F_mass,
        H_mass + scale,
        H_mass + scale,
        H_mass + scale,
        Cl_mass,
        Br_mass,
        H_mass + scale,
    ]
    np.testing.assert_almost_equal(test_masses, ref_masses)


@pytest.fixture()
def arbitrary_transformation():
    # NOTE: test system can probably be simplified; we just need
    # any SingleTopology and conformation
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols = {get_mol_name(mol): mol for mol in read_sdf(path_to_ligand)}

    mol_a = mols["206"]
    mol_b = mols["57"]

    core = _get_core_by_mcs(mol_a, mol_b)
    ff = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, ff)
    conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))
    return st, conf


@pytest.mark.nocuda
def test_jax_transform_intermediate_potential(arbitrary_transformation):
    st, conf = arbitrary_transformation

    def U(x, lam):
        return st.setup_intermediate_state(lam).get_U_fn()(x)

    _ = jax.jit(U)(conf, 0.1)

    confs = jnp.array([conf for _ in range(10)])
    lambdas = jnp.linspace(0, 1, 10)
    _ = jax.vmap(U)(confs, lambdas)
    _ = jax.jit(jax.vmap(U))(confs, lambdas)


@pytest.mark.nocuda
def test_setup_intermediate_state_not_unreasonably_slow(arbitrary_transformation):
    st, _ = arbitrary_transformation
    n_states = 10
    start_time = time.perf_counter()
    for lam in np.linspace(0, 1, n_states):
        _ = st.setup_intermediate_state(lam)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # weak assertion to catch egregious perf issues while being unlikely to raise false positives
    assert elapsed_time / n_states <= 1.0


@pytest.mark.nocuda
def test_setup_intermediate_bonded_term(arbitrary_transformation):
    """Tests that the current vectorized implementation _setup_intermediate_bonded_term is consistent with the previous
    implementation"""
    st, _ = arbitrary_transformation
    interpolate_fn = functools.partial(interpolate_harmonic_bond_params, k_min=0.1, lambda_min=0.0, lambda_max=0.7)

    def setup_intermediate_bonded_term_ref(src_bond, dst_bond, lamb, align_fn, interpolate_fn):
        bond_idxs_and_params = align_fn(
            src_bond.potential.idxs,
            src_bond.params,
            dst_bond.potential.idxs,
            dst_bond.params,
        )

        bond_idxs = []
        bond_params = []

        for idxs, src_params, dst_params in bond_idxs_and_params:
            bond_idxs.append(idxs)
            new_params = interpolate_fn(src_params, dst_params, lamb)
            bond_params.append(new_params)

        return type(src_bond.potential)(np.array(bond_idxs)).bind(jnp.array(bond_params))

    for lamb in np.linspace(0.0, 1.0, 10):
        bonded_ref = setup_intermediate_bonded_term_ref(
            st.src_system.bond, st.dst_system.bond, lamb, align_harmonic_bond_idxs_and_params, interpolate_fn
        )
        bonded_test = st._setup_intermediate_bonded_term(
            st.src_system.bond, st.dst_system.bond, lamb, align_harmonic_bond_idxs_and_params, interpolate_fn
        )

        np.testing.assert_array_equal(bonded_ref.potential.idxs, bonded_test.potential.idxs)
        np.testing.assert_array_equal(bonded_ref.params, bonded_test.params)


@pytest.mark.nocuda
def test_setup_intermediate_nonbonded_term(arbitrary_transformation):
    """Tests that the current vectorized implementation _setup_intermediate_nonbonded_term is consistent with the
    previous implementation"""
    st, _ = arbitrary_transformation

    def setup_intermediate_nonbonded_term_ref(src_nonbonded, dst_nonbonded, lamb, align_fn, interpolate_qlj_fn):
        pair_idxs_and_params = align_fn(
            src_nonbonded.potential.idxs,
            src_nonbonded.params,
            dst_nonbonded.potential.idxs,
            dst_nonbonded.params,
        )

        cutoff = src_nonbonded.potential.cutoff

        pair_idxs = []
        pair_params = []
        for idxs, src_params, dst_params in pair_idxs_and_params:
            src_qlj, src_w = src_params[:3], src_params[3]
            dst_qlj, dst_w = dst_params[:3], dst_params[3]

            if src_qlj == (0, 0, 0):  # i.e. excluded in src state
                new_params = (*dst_qlj, interpolate_w_coord(cutoff, 0, lamb))
            elif dst_qlj == (0, 0, 0):
                new_params = (*src_qlj, interpolate_w_coord(0, cutoff, lamb))
            else:
                new_params = (
                    *interpolate_qlj_fn(jnp.array(src_qlj), jnp.array(dst_qlj), lamb),
                    interpolate_w_coord(src_w, dst_w, lamb),
                )

            pair_idxs.append(idxs)
            pair_params.append(new_params)

        return potentials.NonbondedPairListPrecomputed(
            np.array(pair_idxs), src_nonbonded.potential.beta, src_nonbonded.potential.cutoff
        ).bind(jnp.array(pair_params))

    for lamb in np.linspace(0.0, 1.0, 10):
        nonbonded_ref = setup_intermediate_nonbonded_term_ref(
            st.src_system.nonbonded,
            st.dst_system.nonbonded,
            lamb,
            align_nonbonded_idxs_and_params,
            linear_interpolation,
        )
        nonbonded_test = st._setup_intermediate_nonbonded_term(
            st.src_system.nonbonded,
            st.dst_system.nonbonded,
            lamb,
            align_nonbonded_idxs_and_params,
            linear_interpolation,
        )

        np.testing.assert_array_equal(nonbonded_ref.potential.idxs, nonbonded_test.potential.idxs)
        np.testing.assert_array_equal(nonbonded_ref.params, nonbonded_test.params)


@pytest.mark.nocuda
def test_combine_with_host():
    """Verifies that combine_with_host correctly sets up all of the U functions"""
    mol_a = Chem.MolFromSmiles("BrC1=CC=CC=C1")
    mol_b = Chem.MolFromSmiles("C1=CN=CC=C1F")

    AllChem.EmbedMolecule(mol_a, randomSeed=2022)
    AllChem.EmbedMolecule(mol_b, randomSeed=2022)

    core = np.array([[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    solvent_sys, solvent_conf, _, _ = build_water_system(4.0, ff.water_ff, mols=[mol_a, mol_b])
    host_bps, _ = openmm_deserializer.deserialize_system(solvent_sys, cutoff=1.2)

    st = SingleTopology(mol_a, mol_b, core, ff)
    host_system = st.combine_with_host(convert_bps_into_system(host_bps), 0.5, solvent_conf.shape[0])
    assert set(type(bp.potential) for bp in host_system.get_U_fns()) == {
        potentials.HarmonicBond,
        potentials.HarmonicAngleStable,
        potentials.PeriodicTorsion,
        potentials.NonbondedPairListPrecomputed,
        potentials.Nonbonded,
        potentials.SummedPotential,  # P-L + L-W interactions
        potentials.ChiralAtomRestraint,
        # potentials.ChiralBondRestraint,
        # NOTE: chiral bond restraints excluded
        # This should be updated when chiral restraints are re-enabled.
    }


@pytest.mark.parametrize("precision, rtol, atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
@pytest.mark.parametrize("use_tiny_mol", [True, False])
def test_nonbonded_intra_split(precision, rtol, atol, use_tiny_mol):
    # mol with no intramolecular NB terms and no dihedrals
    if use_tiny_mol:
        mol_a = ligand_from_smiles("S")
        mol_b = ligand_from_smiles("O")
    else:
        with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
            mols = {get_mol_name(mol): mol for mol in read_sdf(path_to_ligand)}
        mol_a = mols["338"]
        mol_b = mols["43"]
    core = _get_core_by_mcs(mol_a, mol_b)

    # split forcefield has different parameters for intramol and intermol terms
    ffs = load_split_forcefields()
    solvent_sys, solvent_conf, solvent_box, solvent_top = build_water_system(4.0, ffs.ref.water_ff, mols=[mol_a, mol_b])
    solvent_conf = minimizer.fire_minimize_host(
        [mol_a, mol_b], HostConfig(solvent_sys, solvent_conf, solvent_box, solvent_conf.shape[0]), ffs.ref
    )
    solvent_bps, _ = openmm_deserializer.deserialize_system(solvent_sys, cutoff=1.2)
    solv_sys = convert_bps_into_system(solvent_bps)

    def get_vacuum_solvent_u_grads(ff, lamb):
        st = SingleTopology(mol_a, mol_b, core, ff)
        ligand_conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b), lamb)
        combined_conf = np.concatenate([solvent_conf, ligand_conf])

        vacuum_system = st.setup_intermediate_state(lamb)
        vacuum_potentials = vacuum_system.get_U_fns()
        val_and_grad_fn = minimizer.get_val_and_grad_fn(vacuum_potentials, solvent_box, precision=precision)
        vacuum_u, vacuum_grad = val_and_grad_fn(ligand_conf)

        solvent_system = st.combine_with_host(solv_sys, lamb, solvent_conf.shape[0])
        solvent_potentials = solvent_system.get_U_fns()
        solv_val_and_grad_fn = minimizer.get_val_and_grad_fn(solvent_potentials, solvent_box, precision=precision)
        solvent_u, solvent_grad = solv_val_and_grad_fn(combined_conf)
        return vacuum_grad, vacuum_u, solvent_grad, solvent_u

    n_lambdas = 3
    for lamb in np.linspace(0, 1, n_lambdas):
        # Compute the grads, potential with the ref ff
        vacuum_grad_ref, vacuum_u_ref, solvent_grad_ref, solvent_u_ref = get_vacuum_solvent_u_grads(ffs.ref, lamb)

        # Compute the grads, potential with the scaled ff
        vacuum_grad_scaled, vacuum_u_scaled, solvent_grad_scaled, solvent_u_scaled = get_vacuum_solvent_u_grads(
            ffs.scaled, lamb
        )

        # Compute the grads, potential with the intermol (ligand-env) scaled ff
        (
            vacuum_grad_env,
            vacuum_u_env,
            solvent_grad_env,
            solvent_u_env,
        ) = get_vacuum_solvent_u_grads(ffs.env, lamb)

        # Compute the expected intermol scaled potential
        expected_env_u = solvent_u_scaled - vacuum_u_scaled + vacuum_u_ref

        # Pad gradients for the solvent
        vacuum_grad_scaled_padded = np.concatenate([np.zeros(solvent_conf.shape), vacuum_grad_scaled])
        vacuum_grad_ref_padded = np.concatenate([np.zeros(solvent_conf.shape), vacuum_grad_ref])
        expected_env_grad = solvent_grad_scaled - vacuum_grad_scaled_padded + vacuum_grad_ref_padded

        # They should be equal
        assert expected_env_u == pytest.approx(solvent_u_env, rel=rtol, abs=atol)
        minimizer.check_force_norm(-expected_env_grad)
        minimizer.check_force_norm(-solvent_grad_env)
        np.testing.assert_allclose(expected_env_grad, solvent_grad_env, rtol=rtol, atol=atol)

        # The vacuum term should be the same as the ref
        assert vacuum_u_env == pytest.approx(vacuum_u_ref, rel=rtol, abs=atol)
        minimizer.check_force_norm(-vacuum_grad_ref)
        minimizer.check_force_norm(-vacuum_grad_env)
        np.testing.assert_allclose(vacuum_grad_ref, vacuum_grad_env, rtol=rtol, atol=atol)


class SingleTopologyRef(SingleTopology):
    def _parameterize_host_guest_nonbonded_ixn(self, lamb, host_nonbonded, _):
        # Parameterize nonbonded potential for the host guest interaction
        num_host_atoms = host_nonbonded.params.shape[0]
        num_guest_atoms = self.get_num_atoms()

        host_params = host_nonbonded.params
        cutoff = host_nonbonded.potential.cutoff

        guest_params = self._get_guest_params(self.ff.q_handle, self.ff.lj_handle, lamb, cutoff)
        combined_nonbonded_params = np.concatenate([host_params, guest_params])

        host_guest_nonbonded_ixn = potentials.NonbondedInteractionGroup(
            num_host_atoms + num_guest_atoms,
            np.arange(num_host_atoms, dtype=np.int32),
            host_nonbonded.potential.beta,
            host_nonbonded.potential.cutoff,
        ).bind(combined_nonbonded_params)

        return host_guest_nonbonded_ixn


@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("lamb", [0.0, 0.5, 1.0])
def test_nonbonded_intra_split_bitwise_identical(precision, lamb):
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols = {get_mol_name(mol): mol for mol in read_sdf(path_to_ligand)}
    mol_a = mols["338"]
    mol_b = mols["43"]
    core = _get_core_by_mcs(mol_a, mol_b)

    ff = Forcefield.load_default()

    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        complex_system, complex_coords, box, _, num_water_atoms = build_protein_system(
            str(path_to_pdb), ff.protein_ff, ff.water_ff
        )
        box += np.diag([0.1, 0.1, 0.1])

    host_bps, host_masses = openmm_deserializer.deserialize_system(complex_system, cutoff=1.2)
    host_system = convert_bps_into_system(host_bps)
    st_ref = SingleTopologyRef(mol_a, mol_b, core, ff)

    combined_ref = st_ref.combine_with_host(host_system, lamb, num_water_atoms)
    ref_potentials = combined_ref.get_U_fns()
    ref_summed = potentials.SummedPotential(
        [bp.potential for bp in ref_potentials], [bp.params for bp in ref_potentials]
    )
    flattened_ref_params = np.concatenate([bp.params.reshape(-1) for bp in ref_potentials])

    st_split = SingleTopology(mol_a, mol_b, core, ff)
    combined_split = st_split.combine_with_host(host_system, lamb, num_water_atoms)
    split_potentials = combined_split.get_U_fns()
    split_summed = potentials.SummedPotential(
        [bp.potential for bp in split_potentials], [bp.params for bp in split_potentials]
    )
    flattened_split_params = np.concatenate([bp.params.reshape(-1) for bp in split_potentials])

    ligand_conf = st_ref.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b), lamb)
    combined_conf = np.concatenate([complex_coords, ligand_conf])

    # Ensure that the du_dx and du_dp are exactly identical, ignore du_dp as shapes are different
    ref_du_dx, _, ref_u = ref_summed.to_gpu(precision).unbound_impl.execute(
        combined_conf, flattened_ref_params, box, True, False, True
    )
    split_du_dx, _, split_u = split_summed.to_gpu(precision).unbound_impl.execute(
        combined_conf, flattened_split_params, box, True, False, True
    )
    np.testing.assert_array_equal(ref_du_dx, split_du_dx)
    np.testing.assert_equal(ref_u, split_u)


@pytest.mark.parametrize("precision, rtol, atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
def test_combine_with_host_split(precision, rtol, atol):
    # test the split P-L and L-W interactions

    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols = {get_mol_name(mol): mol for mol in read_sdf(path_to_ligand)}
    mol_a = mols["338"]
    mol_b = mols["43"]
    core = _get_core_by_mcs(mol_a, mol_b)

    def compute_ref_grad_u(ff: Forcefield, precision, x0, box, lamb, num_water_atoms, host_bps):
        # Use the original code to compute the nb grads and potential
        host_system = convert_bps_into_system(host_bps)
        st = SingleTopologyRef(mol_a, mol_b, core, ff)
        ligand_conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b), lamb)
        num_host_atoms = x0.shape[0] - ligand_conf.shape[0]
        combined_conf = np.concatenate([x0[:num_host_atoms], ligand_conf])

        combined_system = st.combine_with_host(host_system, lamb, num_water_atoms)
        potentials = combined_system.get_U_fns()
        u, grad = minimizer.get_val_and_grad_fn(potentials, box, precision=precision)(combined_conf)
        return grad, u

    def compute_new_grad_u(ff: Forcefield, precision, x0, box, lamb, num_water_atoms, host_bps):
        host_system = convert_bps_into_system(host_bps)
        st = SingleTopology(mol_a, mol_b, core, ff)
        ligand_conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b), lamb)
        num_host_atoms = x0.shape[0] - ligand_conf.shape[0]
        combined_conf = np.concatenate([x0[:num_host_atoms], ligand_conf])

        combined_system = st.combine_with_host(host_system, lamb, num_water_atoms)
        potentials = combined_system.get_U_fns()
        u, grad = minimizer.get_val_and_grad_fn(potentials, box, precision=precision)(combined_conf)
        return grad, u

    def compute_intra_grad_u(ff: Forcefield, precision, x0, box, lamb, num_water_atoms, num_host_atoms):
        st = SingleTopology(mol_a, mol_b, core, ff)
        ligand_conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b), lamb)

        vacuum_system = st.setup_intermediate_state(lamb)
        potentials = vacuum_system.get_U_fns()
        u, grad = minimizer.get_val_and_grad_fn(potentials, box, precision=precision)(ligand_conf)

        # Pad g so it's the same shape as the others
        grad_padded = np.concatenate([np.zeros((num_host_atoms, 3)), grad])
        return grad_padded, u

    def compute_ixn_grad_u(
        ff: Forcefield,
        precision,
        x0,
        box,
        lamb,
        num_water_atoms,
        host_bps,
        water_idxs,
        ligand_idxs,
        protein_idxs,
        is_solvent=False,
    ):
        assert num_water_atoms == len(water_idxs)
        host_system = convert_bps_into_system(host_bps)
        st = SingleTopology(mol_a, mol_b, core, ff)
        ligand_conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b), lamb)
        num_host_atoms = x0.shape[0] - ligand_conf.shape[0]
        combined_conf = np.concatenate([x0[:num_host_atoms], ligand_conf])
        num_total_atoms = combined_conf.shape[0]

        cutoff = host_system.nonbonded.potential.cutoff
        u = potentials.NonbondedInteractionGroup(
            num_total_atoms,
            ligand_idxs,
            host_system.nonbonded.potential.beta,
            cutoff,
            col_atom_idxs=water_idxs if is_solvent else protein_idxs,
        )

        q_handle = ff.q_handle
        lj_handle = ff.lj_handle
        guest_params = st._get_guest_params(q_handle, lj_handle, lamb, cutoff)

        host_params = host_system.nonbonded.params
        combined_nonbonded_params = np.concatenate([host_params, guest_params])
        u_impl = u.bind(combined_nonbonded_params).to_gpu(precision=precision).bound_impl
        return u_impl.execute(combined_conf, box)

    ffs = load_split_forcefields()
    st = SingleTopologyRef(mol_a, mol_b, core, ffs.ref)
    ligand_conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b), 0.0)
    ligand_idxs = np.arange(ligand_conf.shape[0], dtype=np.int32)

    check_split_ixns(
        ligand_conf,
        ligand_idxs,
        precision,
        rtol,
        atol,
        compute_ref_grad_u,
        compute_new_grad_u,
        compute_intra_grad_u,
        compute_ixn_grad_u,
    )


def ligand_from_smiles(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.Compute2DCoords(mol)
    set_mol_name(mol, smiles)
    return mol


def _get_core_by_mcs(mol_a, mol_b):
    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        **DEFAULT_ATOM_MAPPING_KWARGS,
    )

    core = all_cores[0]
    return core


@pytest.mark.nocuda
def test_no_chiral_atom_restraints():
    mol_a = ligand_from_smiles("c1ccccc1")
    mol_b = ligand_from_smiles("c1(I)ccccc1")
    core = _get_core_by_mcs(mol_a, mol_b)

    forcefield = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, forcefield)
    init_conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))
    state = st.setup_intermediate_state(0.1)

    assert len(state.chiral_atom.potential.idxs) == 0
    U = state.get_U_fn()
    _ = U(init_conf)


@pytest.mark.nocuda
def test_no_chiral_bond_restraints():
    mol_a = ligand_from_smiles("C")
    mol_b = ligand_from_smiles("CI")
    core = _get_core_by_mcs(mol_a, mol_b)

    forcefield = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, forcefield)
    init_conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))
    state = st.setup_intermediate_state(0.1)

    assert len(state.chiral_bond.potential.idxs) == 0
    U = state.get_U_fn()
    _ = U(init_conf)


finite_floats = functools.partial(st.floats, allow_nan=False, allow_infinity=False, allow_subnormal=False)

nonzero_force_constants = finite_floats(1e-9, 1e9)

lambdas = finite_floats(0.0, 1.0)


def pairs(elem, unique=False):
    return st.lists(elem, min_size=2, max_size=2, unique=unique).map(tuple)


# https://github.com/python/mypy/issues/12617
lambda_intervals = pairs(finite_floats(1e-9, 1.0 - 1e-9), unique=True).map(sorted)  # type: ignore


@pytest.mark.nocuda
@pytest.mark.parametrize(
    "interpolation_fn",
    [
        linear_interpolation,
        functools.partial(log_linear_interpolation, min_value=0.01),
    ],
)
@given(nonzero_force_constants, lambda_intervals, lambdas)
@seed(2023)
def test_handle_ring_opening_closing_symmetric(interpolation_fn, k, lambda_interval, lam):
    lambda_min, lambda_max = lambda_interval

    # avoid spurious failure due to loss of precision
    assume((lam <= lambda_min) == (lam <= 1 - (1 - lambda_min)))

    f = functools.partial(
        handle_ring_opening_closing,
        interpolation_fn,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
    )

    np.testing.assert_allclose(
        f(0.0, k, lam),
        f(k, 0.0, 1.0 - lam),
        atol=1e-6,
    )


# https://github.com/python/mypy/issues/12617
@pytest.mark.nocuda
@given(nonzero_force_constants, st.lists(lambdas, min_size=3, max_size=3, unique=True).map(sorted))  # type: ignore
@seed(2022)
def test_handle_ring_opening_closing_pin_to_end_states(k, lambdas):
    lam, lambda_min, lambda_max = lambdas
    assert handle_ring_opening_closing(linear_interpolation, 0.0, k, lam, lambda_min, lambda_max) == 0.0

    lambda_min, lambda_max, lam = lambdas
    assert handle_ring_opening_closing(linear_interpolation, 0.0, k, lam, lambda_min, lambda_max) == k


@given(
    nonzero_force_constants,
    nonzero_force_constants,
    nonzero_force_constants,
    lambda_intervals,
)
@seed(2022)
def test_interpolate_harmonic_force_constant(src_k, dst_k, k_min, lambda_interval):
    lambda_min, lambda_max = lambda_interval

    f = functools.partial(
        interpolate_harmonic_force_constant,
        k_min=k_min,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
    )

    assert f(src_k, dst_k, 0.0) == src_k
    assert f(src_k, dst_k, 1.0) == dst_k

    lambdas = np.arange(0.01, 1.0, 0.01)

    # all interpolated values >= k_min
    np.testing.assert_array_less(1.0, f(src_k, dst_k, lambdas) / k_min + 1e-9)

    def assert_nondecreasing(f):
        y = f(lambdas)
        np.testing.assert_array_less(y[:-1] / y[1:], 1.0 + 1e-9)

    k1, k2 = sorted([src_k, dst_k])
    assert_nondecreasing(lambda lam: f(k1, k2, lam))
    assert_nondecreasing(lambda lam: f(k2, k1, 1.0 - lam))


@pytest.mark.nocuda
@given(pairs(nonzero_force_constants, unique=True).filter(lambda ks: np.abs(ks[0] - ks[1]) / ks[1] > 1e-6))
@seed(2022)
def test_interpolate_harmonic_force_constant_sublinear(ks):
    src_k, dst_k = ks
    lambdas = np.arange(0.01, 1.0, 0.01)
    np.testing.assert_array_less(
        interpolate_harmonic_force_constant(src_k, dst_k, lambdas, 1e-12, 0.0, 1.0)
        / linear_interpolation(src_k, dst_k, lambdas),
        1.0,
    )


@pytest.mark.nocuda
def test_interpolate_harmonic_force_constant_jax_transformable():
    _ = jax.jit(interpolate_harmonic_force_constant)(0.0, 1.0, 0.1, 1e-12, 0.0, 1.0)


@pytest.mark.nocuda
def test_cyclic_difference():
    assert cyclic_difference(0, 0, 1) == 0
    assert cyclic_difference(0, 1, 2) == 1  # arbitrary, positive by convention
    assert cyclic_difference(0, 0, 3) == 0
    assert cyclic_difference(0, 1, 3) == 1
    assert cyclic_difference(0, 2, 3) == -1

    # antisymmetric
    assert cyclic_difference(0, 1, 3) == -cyclic_difference(1, 0, 3)
    assert cyclic_difference(0, 2, 3) == -cyclic_difference(2, 0, 3)

    # translation invariant
    assert cyclic_difference(0, 1, 3) == cyclic_difference(-1, 0, 3)
    assert cyclic_difference(0, 4, 8) == cyclic_difference(-2, 2, 8) == cyclic_difference(-4, 0, 8)

    # jittable
    _ = jax.jit(cyclic_difference)(0, 1, 1)


def assert_equal_cyclic(a, b, period):
    def f(x):
        x_mod = x % period
        return np.minimum(x_mod, period - x_mod)

    assert f(a) == f(b)


periods = st.integers(1, int(1e9))
bounded_ints = st.integers(-int(1e9), int(1e9))


@pytest.mark.nocuda
@given(bounded_ints, bounded_ints, periods)
@seed(2022)
def test_cyclic_difference_inverse(a, b, period):
    x = cyclic_difference(a, b, period)
    assert np.abs(x) <= period / 2
    assert_equal_cyclic(a + x, b, period)


@pytest.mark.nocuda
@given(bounded_ints, bounded_ints, periods)
@seed(2022)
def test_cyclic_difference_antisymmetric(a, b, period):
    assert cyclic_difference(a, b, period) + cyclic_difference(b, a, period) == 0


@pytest.mark.nocuda
@given(bounded_ints, bounded_ints, bounded_ints, bounded_ints, periods)
@seed(2022)
def test_cyclic_difference_shift_by_n_periods(a, b, m, n, period):
    assert_equal_cyclic(
        cyclic_difference(a + m * period, b + n * period, period),
        cyclic_difference(a, b, period),
        period,
    )


@pytest.mark.nocuda
@given(bounded_ints, bounded_ints, bounded_ints, periods)
@seed(2022)
def test_cyclic_difference_translation_invariant(a, b, t, period):
    assert_equal_cyclic(
        cyclic_difference(a + t, b + t, period),
        cyclic_difference(a, b, period),
        period,
    )


@pytest.mark.nocuda
@given(pairs(finite_floats()))
@seed(2022)
def test_interpolate_w_coord_valid_at_end_states(end_states):
    a, b = end_states
    f = functools.partial(interpolate_w_coord, a, b)
    assert f(0.0) == a
    assert f(1.0) == b


@pytest.mark.nocuda
def test_interpolate_w_coord_monotonic():
    lambdas = np.linspace(0.0, 1.0, 100)
    ws = interpolate_w_coord(0.0, 1.0, lambdas)
    assert np.all(np.diff(ws) >= 0.0)


@pytest.mark.skip(reason="schedule debug")
@pytest.mark.nocuda
def test_hif2a_plot_force_constants():
    # generate plots of force constants
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols = read_sdf(path_to_ligand)

    pairs = [(mol_a, mol_b) for mol_a in mols for mol_b in mols]

    np.random.seed(2023)
    np.random.shuffle(pairs)
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    # this has been tested for up to 50 random pairs
    for pair_idx, (mol_a, mol_b) in enumerate(pairs[:10]):
        if mol_a.GetProp("_Name") == mol_b.GetProp("_Name"):
            continue

        print("Checking pair", pair_idx, " | ", get_mol_name(mol_a), "->", get_mol_name(mol_b))
        core = _get_core_by_mcs(mol_a, mol_b)
        st = SingleTopology(mol_a, mol_b, core, ff)

        n_windows = 128

        bond_ks = []
        angle_ks = []
        torsion_ks = []
        chiral_atom_ks = []

        xs = np.linspace(0, 1, n_windows)
        for lamb in xs:
            vac_sys = st.setup_intermediate_state(lamb)
            lamb_bond_ks = []
            for k, _ in vac_sys.bond.params:
                lamb_bond_ks.append(k)
            bond_ks.append(lamb_bond_ks)

            lamb_angle_ks = []
            for k, _, _ in vac_sys.angle.params:
                lamb_angle_ks.append(k)
            angle_ks.append(lamb_angle_ks)

            lamb_torsion_ks = []
            for k, _, _ in vac_sys.torsion.params:
                lamb_torsion_ks.append(k)
            torsion_ks.append(lamb_torsion_ks)

            lamb_chiral_atom_ks = []
            for k in vac_sys.chiral_atom.params:
                lamb_chiral_atom_ks.append(k)
            chiral_atom_ks.append(lamb_chiral_atom_ks)

        bond_ks = np.array(bond_ks).T
        angle_ks = np.array(angle_ks).T
        torsion_ks = np.array(torsion_ks).T
        chiral_atom_ks = np.array(chiral_atom_ks).T

        bond_ks /= np.amax(bond_ks, axis=1, keepdims=True)
        angle_ks /= np.amax(angle_ks, axis=1, keepdims=True)
        torsion_ks /= np.amax(torsion_ks, axis=1, keepdims=True)
        chiral_atom_ks /= np.amax(chiral_atom_ks, axis=1, keepdims=True)

        import matplotlib.pyplot as plt

        fig, all_axes = plt.subplots(4, 1, figsize=(1 * 5, 4 * 3))
        fig.tight_layout()

        for v in bond_ks:
            all_axes[0].plot(xs, v)
        all_axes[0].set_title("bond")
        all_axes[0].set_ylabel("fraction of full strength")
        all_axes[0].set_xlabel("lambda")
        all_axes[0].axvline(0.3, ls="--", color="gray")
        all_axes[0].axvline(0.6, ls="--", color="gray")
        all_axes[0].axvline(0.8, ls="--", color="gray")
        all_axes[0].set_ylim(0, 1)

        for v in angle_ks:
            all_axes[1].plot(xs, v)
        all_axes[1].set_title("angle")
        all_axes[1].set_ylabel("fraction of full strength")
        all_axes[1].set_xlabel("lambda")
        all_axes[1].axvline(0.3, ls="--", color="gray")
        all_axes[1].axvline(0.6, ls="--", color="gray")
        all_axes[1].axvline(0.8, ls="--", color="gray")
        all_axes[1].set_ylim(0, 1)

        for v in torsion_ks:
            all_axes[2].plot(xs, v)
        all_axes[2].set_title("torsion")
        all_axes[2].set_ylabel("fraction of full strength")
        all_axes[2].set_xlabel("lambda")
        all_axes[2].axvline(0.3, ls="--", color="gray")
        all_axes[2].axvline(0.6, ls="--", color="gray")
        all_axes[2].axvline(0.8, ls="--", color="gray")
        all_axes[2].set_ylim(0, 1)

        for v in chiral_atom_ks:
            all_axes[3].plot(xs, v)
        all_axes[3].set_title("chiral atom")
        all_axes[3].set_ylabel("fraction of full strength")
        all_axes[3].set_xlabel("lambda")
        all_axes[3].axvline(0.3, ls="--", color="gray")
        all_axes[3].axvline(0.6, ls="--", color="gray")
        all_axes[3].axvline(0.8, ls="--", color="gray")
        all_axes[3].set_ylim(0, 1)

        plt.show()


@pytest.mark.nightly(reason="Test setting up hif2a pairs for single topology.")
@pytest.mark.nocuda
def test_hif2a_pairs_setup_st():
    """
    Test that we can setup all-pairs single topology objects in hif2a.
    """
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols = read_sdf(path_to_ligand)

    pairs = [(mol_a, mol_b) for mol_a in mols for mol_b in mols]
    np.random.seed(2023)
    np.random.shuffle(pairs)
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    for mol_a, mol_b in pairs:
        print(mol_a.GetProp("_Name"), "->", mol_b.GetProp("_Name"))
        core = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)[0]
        SingleTopology(mol_a, mol_b, core, ff)  # Test that this doesn't not throw assertion


@pytest.mark.nocuda
def test_chiral_volume_spiro_failure():
    # test that single topology throws an assertion when morphing
    #
    #    c   c        c   c
    #   / \ / \      / \ / \
    #  c   c   c -> c   c   c
    #   \ . . /      \ / \ /
    #    c   c        c   c
    #    0 restrs    4 restrs
    #
    # (we need at least one restraint to be turned on to enable this)
    mol_a = Chem.MolFromMolBlock(
        """
  Mrv2311 02222400143D

  7  8  0  0  0  0            999 V2000
    1.3547    1.2351    0.0997 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.2235    1.1005    0.7916 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0196   -0.0783    0.0627 C   0  0  2  0  0  0  0  0  0  0  0  0
   -1.1709   -0.2545   -0.6537 O   0  0  0  0  0  0  0  0  0  0  0  0
   -1.3152   -1.3917    0.0257 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.1947   -1.2859    0.7396 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.2208    0.1268   -0.6279 O   0  0  0  0  0  0  0  0  0  0  0  0
  2  3  1  0  0  0  0
  1  2  1  0  0  0  0
  7  3  1  0  0  0  0
  1  7  1  0  0  0  0
  6  3  1  0  0  0  0
  3  4  1  0  0  0  0
  5  6  1  0  0  0  0
  4  5  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """
  Mrv2311 02222400143D

  7  6  0  0  0  0            999 V2000
    1.3547    1.2351    0.0997 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.2235    1.1005    0.7916 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0196   -0.0783    0.0627 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1709   -0.2545   -0.6537 O   0  0  0  0  0  0  0  0  0  0  0  0
   -1.3152   -1.3917    0.0257 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.1947   -1.2859    0.7396 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.2208    0.1268   -0.6279 H   0  0  0  0  0  0  0  0  0  0  0  0
  2  3  1  0  0  0  0
  1  2  1  0  0  0  0
  3  4  1  0  0  0  0
  4  5  1  0  0  0  0
  1  7  1  0  0  0  0
  5  6  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    ff = Forcefield.load_from_file("placeholder_ff.py")
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])

    with pytest.raises(DummyGroupAssignmentError):
        verify_chiral_validity_of_core(mol_a, mol_b, core, ff)

    with pytest.raises(DummyGroupAssignmentError):
        SingleTopology(mol_a, mol_b, core, ff)

    with pytest.raises(DummyGroupAssignmentError):
        verify_chiral_validity_of_core(mol_b, mol_a, core, ff)

    with pytest.raises(DummyGroupAssignmentError):
        SingleTopology(mol_b, mol_a, core, ff)


@pytest.mark.nocuda
def test_chiral_methyl_to_nitrile():
    # test that we do not turn off chiral atom restraints even if some of
    # the angle terms are planar
    #
    #     H        H
    #    .        /
    # N#C-H -> F-C-H
    #    .        \
    #     H        H

    mol_a = Chem.MolFromMolBlock(
        """
  Mrv2311 02232412343D

  5  4  0  0  0  0            999 V2000
    0.4146   -0.0001    0.4976 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0830    0.0001    0.8564 F   0  0  0  0  0  0  0  0  0  0  0  0
    0.5755   -0.0001   -1.0339 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.0830    1.2574    1.0841 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.0830   -1.2574    1.0841 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
  1  2  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """
  Mrv2311 02232412343D

  3  2  0  0  0  0            999 V2000
    0.4146   -0.0001    0.4976 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0830    0.0001    0.8564 N   0  0  0  0  0  0  0  0  0  0  0  0
    0.5755   -0.0001   -1.0339 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  3  1  0  0  0  0
  1  2  3  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    core = np.array([[0, 0], [1, 2]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, ff)
    # chiral force constants should be on for all chiral terms at lambda=0 and lambda=1
    vs_0 = st.setup_intermediate_state(0.0)
    chiral_idxs_0 = vs_0.chiral_atom.potential.idxs
    chiral_params_0 = vs_0.chiral_atom.params
    assert len(chiral_idxs_0) == 4
    assert np.sum(chiral_params_0 == DEFAULT_CHIRAL_ATOM_RESTRAINT_K) == 4
    vs_1 = st.setup_intermediate_state(1.0)

    chiral_idxs_1 = vs_1.chiral_atom.potential.idxs
    chiral_params_1 = vs_1.chiral_atom.params
    assert len(chiral_idxs_0) == len(chiral_idxs_1)
    assert np.sum(chiral_params_1 == DEFAULT_CHIRAL_ATOM_RESTRAINT_K) == 4


@pytest.mark.nocuda
def test_chiral_methyl_to_nitrogen():
    # test that we maintain all 4 chiral idxs when morphing N#N into CH3
    #
    #     H        H
    #    /        /
    # N#N-H -> F-C-H
    #    \        \
    #     H        H
    #
    # (we need at least one restraint to be turned on to enable this)

    mol_a = Chem.MolFromMolBlock(
        """
  Mrv2311 02222400273D

  5  4  0  0  0  0            999 V2000
    0.1976    0.0344    0.3479 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8624    0.0345    0.6018 F   0  0  0  0  0  0  0  0  0  0  0  0
    0.3115    0.0344   -0.7361 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.6707    0.9244    0.7630 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.6707   -0.8555    0.7630 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
  1  2  1  0  0  0  0
M  END
$$$$
""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """
  Mrv2311 02222400253D

  2  1  0  0  0  0            999 V2000
    0.1976    0.0344    0.3479 N   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8624    0.0345    0.6018 N   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  3  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    core = np.array([[0, 0], [4, 1]])

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, ff)

    vs_0 = st.setup_intermediate_state(0.0)
    chiral_idxs_0 = vs_0.chiral_atom.potential.idxs
    chiral_params_0 = vs_0.chiral_atom.params
    assert len(chiral_idxs_0) == 4
    assert np.sum(chiral_params_0 == DEFAULT_CHIRAL_ATOM_RESTRAINT_K) == 4

    vs_1 = st.setup_intermediate_state(1.0)
    chiral_idxs_1 = vs_1.chiral_atom.potential.idxs
    chiral_params_1 = vs_1.chiral_atom.params
    assert len(chiral_idxs_1) == 4
    assert np.sum(chiral_params_1 == DEFAULT_CHIRAL_ATOM_RESTRAINT_K) == 4

    np.testing.assert_array_equal(chiral_idxs_0, chiral_idxs_1)


@pytest.mark.nocuda
def test_chiral_methyl_to_water():
    mol_a = Chem.MolFromMolBlock(
        """
  Mrv2311 02222411113D

  5  4  0  0  0  0            999 V2000
   -1.1951   -0.2262   -0.1811 F   0  0  0  0  0  0  0  0  0  0  0  0
    0.1566   -0.1865    0.0446 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4366    0.8050    0.4004 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.6863   -0.4026   -0.8832 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.4215   -0.9304    0.7960 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  2  4  1  0  0  0  0
  2  5  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """
  Mrv2311 02222411123D

  3  2  0  0  0  0            999 V2000
   -1.1951   -0.2262   -0.1811 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.1566   -0.1865    0.0446 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.4215   -0.9304    0.7960 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    core = np.array([[0, 0], [1, 1], [2, 2]])

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, ff)

    # chiral force constants should be on for all chiral terms at lambda=0 and lambda=1
    vs_0 = st.setup_intermediate_state(0.0)
    chiral_idxs_0 = vs_0.chiral_atom.potential.idxs
    chiral_params_0 = vs_0.chiral_atom.params
    assert len(chiral_idxs_0) == 4
    assert np.sum(chiral_params_0 == DEFAULT_CHIRAL_ATOM_RESTRAINT_K) == 4
    vs_1 = st.setup_intermediate_state(1.0)

    chiral_idxs_1 = vs_1.chiral_atom.potential.idxs
    chiral_params_1 = vs_1.chiral_atom.params
    assert len(chiral_idxs_0) == len(chiral_idxs_1)
    assert np.sum(chiral_params_1 == DEFAULT_CHIRAL_ATOM_RESTRAINT_K) == 4


@pytest.mark.nocuda
def test_chiral_methyl_to_ammonia():
    mol_a = Chem.MolFromMolBlock(
        """
  Mrv2311 02232411003D

  5  4  0  0  0  0            999 V2000
    0.0402    0.0126    0.1841 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.2304   -0.7511    0.9383 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.8502    0.0126   -0.5452 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0173    0.9900    0.6632 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9024   -0.2011   -0.3198 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """
  Mrv2311 02232411003D

  4  3  0  0  0  0            999 V2000
    0.0402    0.0126    0.1841 N   0  0  0  0  0  0  0  0  0  0  0  0
    0.2304   -0.7511    0.9383 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0173    0.9900    0.6632 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9024   -0.2011   -0.3198 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, ff)

    # chiral force constants should be on for all chiral terms at lambda=0 and lambda=1
    vs_0 = st.setup_intermediate_state(0.0)
    chiral_idxs_0 = vs_0.chiral_atom.potential.idxs
    chiral_params_0 = vs_0.chiral_atom.params
    assert len(chiral_idxs_0) == 4
    assert np.sum(chiral_params_0 == DEFAULT_CHIRAL_ATOM_RESTRAINT_K) == 4
    vs_1 = st.setup_intermediate_state(1.0)

    # Note that NH3 is categorized as achiral
    chiral_idxs_1 = vs_1.chiral_atom.potential.idxs
    chiral_params_1 = vs_1.chiral_atom.params
    assert len(chiral_idxs_0) == len(chiral_idxs_1)
    assert np.sum(chiral_params_1 == DEFAULT_CHIRAL_ATOM_RESTRAINT_K) == 3


@pytest.mark.nocuda
def test_chiral_core_ring_opening():
    # test that chiral restraints are maintained for dummy atoms when we open/close a ring,
    # at lambda=0, all 7 chiral restraints are turned on, but at lambda=1
    # only 4 chiral restraints are turned on.

    mol_a = Chem.MolFromMolBlock(
        """
  Mrv2311 02222400433D

  6  6  0  0  0  0            999 V2000
   -0.2397    1.3763    0.4334 C   0  0  2  0  0  0  0  0  0  0  0  0
    0.2664   -0.0682    0.6077 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.8332    1.6232   -0.6421 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.3412    0.1809   -0.4674 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0336    1.9673    1.3258 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2364    1.3849   -0.0078 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  3  1  0  0  0  0
  3  4  1  0  0  0  0
  1  2  1  0  0  0  0
  4  2  1  0  0  0  0
  1  5  1  0  0  0  0
  1  6  1  0  0  0  0
M  END
$$$$
""",
        removeHs=False,
    )  # closed ring

    mol_b = Chem.MolFromMolBlock(
        """
  Mrv2311 02222400463D

  7  6  0  0  0  0            999 V2000
   -0.2397    1.3763    0.4334 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.2664   -0.0682    0.6077 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.8332    1.6232   -0.6421 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.3412    0.1809   -0.4674 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0336    1.9673    1.3258 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2364    1.3849   -0.0078 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0787   -0.1553    0.4334 H   0  0  0  0  0  0  0  0  0  0  0  0
  3  4  1  0  0  0  0
  1  3  1  0  0  0  0
  1  5  1  0  0  0  0
  1  6  1  0  0  0  0
  1  7  1  0  0  0  0
  4  2  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )  # open ring

    # map everything except a single hydrogen at the end
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])

    # chiral force constants should be on for all 7 chiral
    # terms at lambda=0
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, ff)
    vs_0 = st.setup_intermediate_state(0.0)
    chiral_idxs_0 = vs_0.chiral_atom.potential.idxs
    chiral_params_0 = vs_0.chiral_atom.params
    assert len(chiral_idxs_0) == 7
    assert np.sum(chiral_params_0 == DEFAULT_CHIRAL_ATOM_RESTRAINT_K) == 7
    vs_1 = st.setup_intermediate_state(1.0)

    # chiral force constants should be on for all 4 of the 7
    # chiral terms at lambda=1
    chiral_idxs_1 = vs_1.chiral_atom.potential.idxs
    chiral_params_1 = vs_1.chiral_atom.params
    assert len(chiral_idxs_0) == len(chiral_idxs_1)

    assert np.sum(chiral_params_1 == 0) == 3
    assert np.sum(chiral_params_1 == DEFAULT_CHIRAL_ATOM_RESTRAINT_K) == 4


def permute_atom_indices(mol_a, mol_b, core, seed):
    """Randomly permute atom indices in mol_a, mol_b independently, and update core"""
    rng = np.random.default_rng(seed)

    perm_a = rng.permutation(mol_a.GetNumAtoms())
    perm_b = rng.permutation(mol_b.GetNumAtoms())

    # RenumberAtoms takes inverse permutations
    # e.g. [3, 2, 0, 1] means atom 3 in the original mol will be atom 0 in the new one
    inv_perm_a = np.argsort(perm_a)
    inv_perm_b = np.argsort(perm_b)
    mol_a = Chem.RenumberAtoms(mol_a, inv_perm_a.tolist())
    mol_b = Chem.RenumberAtoms(mol_b, inv_perm_b.tolist())

    core = np.array(core)
    core[:, 0] = perm_a[core[:, 0]]
    core[:, 1] = perm_b[core[:, 1]]

    return mol_a, mol_b, core


@pytest.mark.parametrize("seed", [2024, 2025])
def test_chiral_core_bond_breaking_raises_error(seed):
    """Test that we raise assertions for molecules that cannot generate valid dummy-group anchor assignments. In
    particular, we break two core-core bonds under an identity mapping (with no dummy atoms).
    """
    mol_a = Chem.MolFromMolBlock(
        """
  Mrv2311 07312412093D

  6  7  0  0  0  0            999 V2000
    1.7504    0.2003    1.2663 N   0  0  2  0  0  0  0  0  0  0  0  0
    0.8845   -0.7367    0.7929 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0656    0.1573    0.2634 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.6778    2.1410    1.4645 F   0  0  0  0  0  0  0  0  0  0  0  0
    0.9965    1.1853    0.5313 C   0  0  1  0  0  0  0  0  0  0  0  0
    2.3507    0.8788    0.2466 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  5  4  1  0  0  0  0
  5  1  1  0  0  0  0
  2  3  1  0  0  0  0
  6  1  1  0  0  0  0
  3  5  1  0  0  0  0
  5  6  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """
  Mrv2311 07312412083D

  6  5  0  0  0  0            999 V2000
    1.7504    0.2003    1.2663 N   0  0  2  0  0  0  0  0  0  0  0  0
    0.8845   -0.7367    0.7929 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0656    0.1573    0.2634 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.6778    2.1410    1.4645 F   0  0  0  0  0  0  0  0  0  0  0  0
    0.9965    1.1853    0.5313 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.3507    0.8788    0.2466 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  5  4  1  0  0  0  0
  5  1  1  0  0  0  0
  6  1  1  0  0  0  0
  2  3  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    ff = Forcefield.load_default()
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])

    mol_a, mol_b, core = permute_atom_indices(mol_a, mol_b, core, seed)

    with pytest.raises(DummyGroupAssignmentError):
        verify_chiral_validity_of_core(mol_a, mol_b, core, ff)


@pytest.mark.parametrize("seed", [2024, 2025])
def test_chiral_bond_breaking_1_core_1_dummy(seed):
    mol_a = Chem.MolFromMolBlock(
        """lhs
                    3D
 Structure written by MMmdl.
 15 16  0  0  1  0            999 V2000
    2.1455    1.6402   -0.1409 O   0  0  0  0  0  0
   -0.0529   -0.7121    0.3721 C   0  0  0  0  0  0
   -1.4286   -0.4301   -0.2680 C   0  0  0  0  0  0
   -0.9221    0.8962   -0.8723 C   0  0  0  0  0  0
    1.7163    0.6542   -1.0827 C   0  0  0  0  0  0
    0.4519    0.5990   -0.2512 C   0  0  0  0  0  0
    1.0309    1.5756    0.6040 O   0  0  0  0  0  0
   -0.0750   -0.6889    1.4650 H   0  0  0  0  0  0
    0.4285   -1.6073   -0.0287 H   0  0  0  0  0  0
   -2.2238   -0.2880    0.4685 H   0  0  0  0  0  0
   -1.6971   -1.1659   -1.0311 H   0  0  0  0  0  0
   -1.4047    1.7800   -0.4452 H   0  0  0  0  0  0
   -0.9239    0.8937   -1.9651 H   0  0  0  0  0  0
    1.5863    1.0639   -2.0863 H   0  0  0  0  0  0
    2.3266   -0.2504   -1.0468 H   0  0  0  0  0  0
  6  2  1  0  0  0
  6  4  1  0  0  0
  6  5  1  0  0  0
  6  7  1  0  0  0
  2  3  1  0  0  0
  2  8  1  0  0  0
  2  9  1  0  0  0
  3  4  1  0  0  0
  3 10  1  0  0  0
  3 11  1  0  0  0
  4 12  1  0  0  0
  4 13  1  0  0  0
  5  1  1  0  0  0
  5 14  1  0  0  0
  5 15  1  0  0  0
  1  7  1  0  0  0
M  END""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """rhs
                    3D
 Structure written by MMmdl.
 12 11  0  0  1  0            999 V2000
    0.4102    0.4907   -0.1997 O   0  0  0  0  0  0
   -0.0529   -0.7121    0.3721 C   0  0  0  0  0  0
   -1.4286   -0.4301   -0.2680 C   0  0  0  0  0  0
   -1.0708    0.5068   -0.6949 H   0  0  0  0  0  0
    1.5875    0.5421   -0.9739 C   0  0  0  0  0  0
    1.9148    1.2939   -0.2558 H   0  0  0  0  0  0
   -0.0750   -0.6889    1.4650 H   0  0  0  0  0  0
    0.4285   -1.6073   -0.0287 H   0  0  0  0  0  0
   -2.2238   -0.2880    0.4685 H   0  0  0  0  0  0
   -1.6971   -1.1659   -1.0311 H   0  0  0  0  0  0
    1.4575    0.9518   -1.9775 H   0  0  0  0  0  0
    2.1978   -0.3625   -0.9380 H   0  0  0  0  0  0
  1  2  1  0  0  0
  1  5  1  0  0  0
  2  3  1  0  0  0
  2  7  1  0  0  0
  2  8  1  0  0  0
  3  4  1  0  0  0
  3  9  1  0  0  0
  3 10  1  0  0  0
  5  6  1  0  0  0
  5 11  1  0  0  0
  5 12  1  0  0  0
M  END
""",
        removeHs=False,
    )

    # need to generate SC charges on this mol - am1 fails
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    core = np.array(
        [[5, 0], [1, 1], [2, 2], [3, 3], [4, 4], [0, 5], [7, 6], [8, 7], [9, 8], [10, 9], [13, 10], [14, 11]]
    )

    mol_a, mol_b, core = permute_atom_indices(mol_a, mol_b, core, seed)

    # from timemachine.fe.utils import plot_atom_mapping_grid

    # print("core", core)
    # res = plot_atom_mapping_grid(mol_a, mol_b, core)
    # fpath = f"atom_mapping.svg"
    # print("core mapping written to", fpath)

    # with open(fpath, "w") as fh:
    #     fh.write(res)

    # should not raise an assertion
    verify_chiral_validity_of_core(mol_a, mol_b, core, ff)
