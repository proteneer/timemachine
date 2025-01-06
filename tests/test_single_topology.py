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
from hypothesis import event, given, seed
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine import potentials
from timemachine.constants import (
    DEFAULT_ATOM_MAPPING_KWARGS,
    DEFAULT_CHIRAL_ATOM_RESTRAINT_K,
    DEFAULT_CHIRAL_BOND_RESTRAINT_K,
    NBParamIdx,
)
from timemachine.fe import atom_mapping, single_topology
from timemachine.fe.dummy import MultipleAnchorWarning, canonicalize_bond
from timemachine.fe.free_energy import HostConfig
from timemachine.fe.interpolate import align_nonbonded_idxs_and_params, linear_interpolation
from timemachine.fe.single_topology import (
    AtomMapMixin,
    ChargePertubationError,
    CoreBondChangeWarning,
    SingleTopology,
    assert_default_system_constraints,
    canonicalize_bonds,
    canonicalize_chiral_atom_idxs,
    canonicalize_improper_idxs,
    cyclic_difference,
    interpolate_w_coord,
    setup_dummy_interactions_from_ff,
)
from timemachine.fe.system import convert_bps_into_system, minimize_scipy, simulate_system
from timemachine.fe.utils import get_mol_name, get_romol_conf, read_sdf, read_sdf_mols_by_name, set_mol_name
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
    mol = ligand_from_smiles("c1ccccc1O", seed=2022)

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
    mol = ligand_from_smiles("C", seed=2022)

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

    mol_a = ligand_from_smiles("CC", seed=2022)
    mol_b = ligand_from_smiles("c1ccc1", seed=2022)

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
    mol_a = ligand_from_smiles("Cc1cc[nH]c1", seed=2022)
    mol_b = ligand_from_smiles("C[n+]1cc[nH]c1", seed=2022)

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


def assert_improper_idxs_are_canonical(all_idxs):
    for idxs in all_idxs:
        np.testing.assert_array_equal(idxs, canonicalize_improper_idxs(idxs))


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
            assert bond_idxs_are_canonical(system.proper.potential.idxs)
            assert_improper_idxs_are_canonical(system.improper.potential.idxs)
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
    improper_idxs = [(0, 5, 1, 3), (1, 5, 3, 0), (3, 5, 0, 1)]

    for idxs in improper_idxs:
        # we should do nothing here.
        assert idxs == canonicalize_improper_idxs(idxs)

    # these are in the ccw rotation set
    assert canonicalize_improper_idxs((1, 5, 0, 3)) == (1, 5, 3, 0)
    assert canonicalize_improper_idxs((3, 5, 1, 0)) == (3, 5, 0, 1)
    assert canonicalize_improper_idxs((0, 5, 3, 1)) == (0, 5, 1, 3)


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
        mols = read_sdf_mols_by_name(path_to_ligand)

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


# @pytest.mark.nocuda
# def test_setup_intermediate_bonded_term(arbitrary_transformation):
#     """Tests that the current vectorized implementation _setup_intermediate_bonded_term is consistent with the previous
#     implementation"""
#     st, _ = arbitrary_transformation
#     interpolate_fn = functools.partial(interpolate_harmonic_bond_params, k_min=0.1, lambda_min=0.0, lambda_max=0.7)

#     def setup_intermediate_bonded_term_ref(src_bond, dst_bond, lamb, align_fn, interpolate_fn):
#         bond_idxs_and_params = align_fn(
#             src_bond.potential.idxs,
#             src_bond.params,
#             dst_bond.potential.idxs,
#             dst_bond.params,
#         )

#         bond_idxs = []
#         bond_params = []

#         for idxs, src_params, dst_params in bond_idxs_and_params:
#             bond_idxs.append(idxs)
#             new_params = interpolate_fn(src_params, dst_params, lamb)
#             bond_params.append(new_params)

#         return type(src_bond.potential)(np.array(bond_idxs)).bind(jnp.array(bond_params))

#     for lamb in np.linspace(0.0, 1.0, 10):
#         bonded_ref = setup_intermediate_bonded_term_ref(
#             st.src_system.bond, st.dst_system.bond, lamb, align_harmonic_bond_idxs_and_params, interpolate_fn
#         )
#         bonded_test = st._setup_intermediate_bonded_term(
#             st.src_system.bond, st.dst_system.bond, lamb, align_harmonic_bond_idxs_and_params, interpolate_fn
#         )

#         np.testing.assert_array_equal(bonded_ref.potential.idxs, bonded_test.potential.idxs)
#         np.testing.assert_array_equal(bonded_ref.params, bonded_test.params)


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
    mol_a = ligand_from_smiles("BrC1=CC=CC=C1", seed=2022)
    mol_b = ligand_from_smiles("C1=CN=CC=C1F", seed=2022)

    core = np.array([[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    solvent_sys, solvent_conf, _, top = build_water_system(4.0, ff.water_ff, mols=[mol_a, mol_b])
    host_bps, _ = openmm_deserializer.deserialize_system(solvent_sys, cutoff=1.2)

    st = SingleTopology(mol_a, mol_b, core, ff)
    host_system = st.combine_with_host(convert_bps_into_system(host_bps), 0.5, solvent_conf.shape[0], ff, top)
    assert set(type(bp.potential) for bp in host_system.get_U_fns()) == {
        potentials.HarmonicBond,
        potentials.HarmonicAngleStable,
        potentials.PeriodicTorsion,
        potentials.NonbondedPairListPrecomputed,
        potentials.Nonbonded,
        potentials.NonbondedInteractionGroup,  # L-P + L-W interactions
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
        # Align the mols that the heavy atom has a common position
        Chem.rdMolAlign.AlignMol(mol_a, mol_b, atomMap=[(0, 0)])
    else:
        with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
            mols = read_sdf_mols_by_name(path_to_ligand)
        mol_a = mols["338"]
        mol_b = mols["43"]
    core = _get_core_by_mcs(mol_a, mol_b)

    # split forcefield has different parameters for intramol and intermol terms
    ffs = load_split_forcefields()
    solvent_sys, solvent_conf, solvent_box, solvent_top = build_water_system(4.0, ffs.ref.water_ff, mols=[mol_a, mol_b])
    solvent_box += np.eye(3) * 0.1
    solvent_conf = minimizer.fire_minimize_host(
        [mol_a, mol_b], HostConfig(solvent_sys, solvent_conf, solvent_box, solvent_conf.shape[0], solvent_top), ffs.ref
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

        solvent_system = st.combine_with_host(solv_sys, lamb, solvent_conf.shape[0], ff, solvent_top)
        solvent_potentials = solvent_system.get_U_fns()
        solv_val_and_grad_fn = minimizer.get_val_and_grad_fn(solvent_potentials, solvent_box, precision=precision)
        solvent_u, solvent_grad = solv_val_and_grad_fn(combined_conf)
        return vacuum_grad, vacuum_u, solvent_grad, solvent_u

    n_lambdas = 3
    for lamb in np.linspace(0, 1, n_lambdas):
        # Compute the grads, potential with the ref ff
        vacuum_grad_ref, vacuum_u_ref, solvent_grad_ref, solvent_u_ref = get_vacuum_solvent_u_grads(ffs.ref, lamb)
        minimizer.check_force_norm(-vacuum_grad_ref)
        minimizer.check_force_norm(-solvent_grad_ref)

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
    def _parameterize_host_guest_nonbonded_ixn(self, lamb, host_nonbonded, *_):
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
        mols = read_sdf_mols_by_name(path_to_ligand)
    mol_a = mols["338"]
    mol_b = mols["43"]
    core = _get_core_by_mcs(mol_a, mol_b)

    ff = Forcefield.load_default()

    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        complex_system, complex_coords, box, complex_top, num_water_atoms = build_protein_system(
            str(path_to_pdb), ff.protein_ff, ff.water_ff
        )
        box += np.diag([0.1, 0.1, 0.1])

    host_bps, host_masses = openmm_deserializer.deserialize_system(complex_system, cutoff=1.2)
    host_system = convert_bps_into_system(host_bps)
    st_ref = SingleTopologyRef(mol_a, mol_b, core, ff)

    combined_ref = st_ref.combine_with_host(host_system, lamb, num_water_atoms, ff, complex_top)
    ref_potentials = combined_ref.get_U_fns()
    ref_summed = potentials.SummedPotential(
        [bp.potential for bp in ref_potentials], [bp.params for bp in ref_potentials]
    )
    flattened_ref_params = np.concatenate([bp.params.reshape(-1) for bp in ref_potentials])

    st_split = SingleTopology(mol_a, mol_b, core, ff)
    combined_split = st_split.combine_with_host(host_system, lamb, num_water_atoms, ff, complex_top)
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
        mols = read_sdf_mols_by_name(path_to_ligand)
    mol_a = mols["338"]
    mol_b = mols["43"]
    core = _get_core_by_mcs(mol_a, mol_b)

    def compute_ref_grad_u(ff: Forcefield, precision, x0, box, lamb, num_water_atoms, host_bps, omm_topology):
        # Use the original code to compute the nb grads and potential
        host_system = convert_bps_into_system(host_bps)
        st = SingleTopologyRef(mol_a, mol_b, core, ff)
        ligand_conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b), lamb)
        num_host_atoms = x0.shape[0] - ligand_conf.shape[0]
        combined_conf = np.concatenate([x0[:num_host_atoms], ligand_conf])

        combined_system = st.combine_with_host(host_system, lamb, num_water_atoms, ff, omm_topology)
        potentials = combined_system.get_U_fns()
        u, grad = minimizer.get_val_and_grad_fn(potentials, box, precision=precision)(combined_conf)
        return grad, u

    def compute_new_grad_u(ff: Forcefield, precision, x0, box, lamb, num_water_atoms, host_bps, omm_topology):
        host_system = convert_bps_into_system(host_bps)
        st = SingleTopology(mol_a, mol_b, core, ff)
        ligand_conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b), lamb)
        num_host_atoms = x0.shape[0] - ligand_conf.shape[0]
        combined_conf = np.concatenate([x0[:num_host_atoms], ligand_conf])

        combined_system = st.combine_with_host(host_system, lamb, num_water_atoms, ff, omm_topology)
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
        omm_topology,
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

        host_ixn_params = host_system.nonbonded.params.copy()
        if not is_solvent and ff.env_bcc_handle is not None:  # protein
            env_bcc_h = ff.env_bcc_handle.get_env_handle(omm_topology, ff)
            host_ixn_params[:, NBParamIdx.Q_IDX] = env_bcc_h.parameterize(ff.env_bcc_handle.params)

        combined_nonbonded_params = np.concatenate([host_ixn_params, guest_params])
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


def ligand_from_smiles(smiles: str, seed: int = 2024) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
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


def pairs(elem, unique=False):
    return st.lists(elem, min_size=2, max_size=2, unique=unique).map(tuple)


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


def get_vacuum_system_and_conf(mol_a, mol_b, core, lamb):
    ff = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, ff)
    conf_a = get_romol_conf(mol_a)
    conf_b = get_romol_conf(mol_b)
    conf = st.combine_confs(conf_a, conf_b, lamb)
    return st.setup_intermediate_state(lamb), conf


def _assert_consistent_hamiltonian_term_impl(fwd_bonded_term, rev_bonded_term, rev_kv, canon_fn):
    canonical_map = dict()
    for fwd_idxs, fwd_params in zip(fwd_bonded_term.potential.idxs, fwd_bonded_term.params):
        fwd_key = canon_fn(fwd_idxs, fwd_params)
        canonical_map[fwd_key] = [fwd_params, None]

    for rev_idxs, rev_params in zip(rev_bonded_term.potential.idxs, rev_bonded_term.params):
        rev_key = canon_fn([rev_kv[x] for x in rev_idxs], rev_params)
        canonical_map[rev_key][1] = rev_params

    for fwd_params, rev_params in canonical_map.values():
        np.testing.assert_allclose(fwd_params, rev_params)


def _assert_u_and_grad_consistent(u_fwd, u_rev, x_fwd, fused_map, canon_fn):
    # test that the definition of the hamiltonian, the energies, and the forces are all consistent
    rev_kv = dict()
    fwd_kv = dict()
    for x, y in fused_map:
        fwd_kv[x] = y
        rev_kv[y] = x
    x_rev = np.zeros_like(x_fwd)
    for atom_idx, xyz in enumerate(x_fwd):
        x_rev[fwd_kv[atom_idx]] = xyz

    # check hamiltonian
    _assert_consistent_hamiltonian_term_impl(u_fwd, u_rev, rev_kv, canon_fn)

    # check energies and forces
    box = 100.0 * np.eye(3)
    np.testing.assert_allclose(u_fwd(x_fwd, box), u_rev(x_rev, box))
    fwd_bond_grad_fn = jax.grad(u_fwd)
    rev_bond_grad_fn = jax.grad(u_rev)
    np.testing.assert_allclose(
        fwd_bond_grad_fn(x_fwd, box)[fused_map[:, 0]], rev_bond_grad_fn(x_rev, box)[fused_map[:, 1]]
    )


def _get_fused_map(mol_a, mol_b, core):
    amm_fwd = AtomMapMixin(mol_a, mol_b, core)
    amm_rev = AtomMapMixin(mol_b, mol_a, core[:, ::-1])
    fused_map = np.concatenate(
        [
            np.array([[x, y] for x, y in zip(amm_fwd.a_to_c, amm_rev.b_to_c)], dtype=np.int32).reshape(-1, 2),
            np.array(
                [[x, y] for x, y in zip(amm_fwd.b_to_c, amm_rev.a_to_c) if x not in core[:, 0] and y not in core[:, 1]],
                dtype=np.int32,
            ).reshape(-1, 2),
        ]
    )
    return fused_map


def assert_symmetric_interpolation(mol_a, mol_b, core):
    """
    Assert that the Single Topology interpolation code is symmetric, i.e. ST(mol_a, mol_b, lamb) == ST(mol_b, mol_a, 1-lamb)

    Where for each of the bond, angle, proper torsion, improper torsion, nonbonded, terms
        - the idxs, params are identical under atom-mapping + canonicalization
        - u_fwd, u_rev for an arbitrary conformation is identical under atom mapping
        - grad_fwd, grad_rev for an arbitrary conformation is identical under atom mapping

    """
    ff = Forcefield.load_default()
    # map atoms in the combined mol_ab to the atoms in the combined mol_ba
    fused_map = _get_fused_map(mol_a, mol_b, core)

    st_fwd = SingleTopology(mol_a, mol_b, core, ff)
    st_rev = SingleTopology(mol_b, mol_a, core[:, ::-1], ff)
    conf_a = get_romol_conf(mol_a)
    conf_b = get_romol_conf(mol_b)
    test_conf = st_fwd.combine_confs(conf_a, conf_b, 0)

    seed = 2024
    np.random.seed(seed)
    lambda_schedule = np.concatenate([np.linspace(0, 1, 12), np.random.rand(10)])

    for lamb in lambda_schedule:
        sys_fwd = st_fwd.setup_intermediate_state(lamb)
        sys_rev = st_rev.setup_intermediate_state(1 - lamb)

        assert_default_system_constraints(sys_fwd)
        assert_default_system_constraints(sys_rev)

        _assert_u_and_grad_consistent(
            sys_fwd.bond, sys_rev.bond, test_conf, fused_map, canon_fn=lambda idxs, _: tuple(canonicalize_bond(idxs))
        )

        _assert_u_and_grad_consistent(
            sys_fwd.angle, sys_rev.angle, test_conf, fused_map, canon_fn=lambda idxs, _: tuple(canonicalize_bond(idxs))
        )

        # for propers, we format the phase as a 5 decimal string to guard against loss of precision
        _assert_u_and_grad_consistent(
            sys_fwd.proper,
            sys_rev.proper,
            test_conf,
            fused_map,
            canon_fn=lambda idxs, params: tuple([*canonicalize_bond(idxs), f"{params[1]:.5f}", int(round(params[2]))]),
        )

        _assert_u_and_grad_consistent(
            sys_fwd.improper,
            sys_rev.improper,
            test_conf,
            fused_map,
            canon_fn=lambda idxs, _: tuple(canonicalize_improper_idxs(idxs)),
        )

        _assert_u_and_grad_consistent(
            sys_fwd.chiral_atom,
            sys_rev.chiral_atom,
            test_conf,
            fused_map,
            canon_fn=lambda idxs, _: tuple(canonicalize_chiral_atom_idxs(np.array([idxs]))[0]),  # fn assumes ndim=2
        )

        _assert_u_and_grad_consistent(
            sys_fwd.nonbonded,
            sys_rev.nonbonded,
            test_conf,
            fused_map,
            canon_fn=lambda idxs, _: tuple(canonicalize_bond(idxs)),
        )


@pytest.mark.nocuda
def test_hif2a_end_state_symmetry_unit_test():
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols = read_sdf(path_to_ligand)

    mol_a = mols[0]
    mol_b = mols[1]
    core = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)[0]
    assert_symmetric_interpolation(mol_a, mol_b, core)


@pytest.mark.nocuda
@pytest.mark.nightly(reason="slow")
def test_hif2a_end_state_symmetry_nightly_test():
    """
    Test that end-states are symmetric for a large number of random pairs
    """
    seed = 2029
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols = read_sdf(path_to_ligand)
    pairs = [(mol_a, mol_b) for mol_a in mols for mol_b in mols]
    np.random.seed(seed)
    np.random.shuffle(pairs)
    for mol_a, mol_b in pairs[:25]:
        print("testing", mol_a.GetProp("_Name"), "->", mol_b.GetProp("_Name"))
        core = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)[0]
        assert_symmetric_interpolation(mol_a, mol_b, core)
