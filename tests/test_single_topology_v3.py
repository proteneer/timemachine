import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(multiprocessing.cpu_count())
import jax

jax.config.update("jax_enable_x64", True)

# test that end-states are setup correctly in single topology calculations.
import functools
import os
from importlib import resources

import jax.numpy as jnp
import numpy as np
import pytest
from jax import vmap
from jax.experimental.checkify import checkify
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine.constants import DEFAULT_FF
from timemachine.fe import atom_mapping, single_topology_v3
from timemachine.fe.interpolate import linear_interpolation
from timemachine.fe.single_topology_v3 import (
    ChargePertubationError,
    CoreBondChangeWarning,
    MultipleAnchorWarning,
    SingleTopologyV3,
    canonicalize_improper_idxs,
    cyclic_difference,
    handle_ring_opening_closing,
    interpolate_harmonic_force_constant,
    setup_dummy_interactions_from_ff,
)
from timemachine.fe.system import convert_bps_into_system, minimize_scipy, simulate_system
from timemachine.fe.utils import get_mol_name, get_romol_conf
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
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


def test_charge_perturbation_is_invalid():
    mol_a = Chem.AddHs(Chem.MolFromSmiles("Cc1cc[nH]c1"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("C[n+]1cc[nH]c1"))

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    core = np.zeros((mol_a.GetNumAtoms(), 2))
    core[:, 0] = np.arange(core.shape[0])
    core[:, 1] = core[:, 0]

    with pytest.raises(ChargePertubationError) as e:
        SingleTopologyV3(mol_a, mol_b, core, ff)
    assert str(e.value) == "mol a and mol b don't have the same charge: a: 0 b: 1"


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
        core_pairs, _ = atom_mapping.get_core_with_alignment(mol_a, mol_b, threshold=mcs_threshold)
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

    _ = jax.jit(checkify(U))(conf, 0.1)

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
    # Expect there to be 5 functions, excluding the chiral bond and chiral atom restraints
    # This should be updated when chiral restraints are re-enabled.
    assert len(host_system.get_U_fns()) == 5


def ligand_from_smiles(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.Compute2DCoords(mol)
    return mol


def get_core_by_mcs(mol_a, mol_b, mcs_threshold=2.0):
    mcs_result = atom_mapping.mcs(mol_a, mol_b, threshold=mcs_threshold, conformer_aware=False)
    query_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    return atom_mapping.get_core_by_mcs(mol_a, mol_b, query_mol, threshold=mcs_threshold)


def test_no_chiral_atom_restraints():
    mol_a = ligand_from_smiles("c1ccccc1")
    mol_b = ligand_from_smiles("c1(I)ccccc1")
    core = get_core_by_mcs(mol_a, mol_b)

    forcefield = Forcefield.load_from_file(DEFAULT_FF)
    st = SingleTopologyV3(mol_a, mol_b, core, forcefield)
    init_conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))
    state = st.setup_intermediate_state(0.1)

    assert len(state.chiral_atom.get_idxs()) == 0
    U = state.get_U_fn()
    _ = U(init_conf)


def test_no_chiral_bond_restraints():
    mol_a = ligand_from_smiles("C")
    mol_b = ligand_from_smiles("CI")
    core = get_core_by_mcs(mol_a, mol_b)

    forcefield = Forcefield.load_from_file(DEFAULT_FF)
    st = SingleTopologyV3(mol_a, mol_b, core, forcefield)
    init_conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))
    state = st.setup_intermediate_state(0.1)

    assert len(state.chiral_bond.get_idxs()) == 0
    U = state.get_U_fn()
    _ = U(init_conf)


def assert_interpolation_valid(f, a, b):
    np.testing.assert_allclose(f(a, b, 0.0), a)
    np.testing.assert_allclose(f(a, b, 1.0), b)

    rng = np.random.default_rng(2022)
    lambdas = rng.uniform(0, 1, (30,))
    ys = vmap(functools.partial(f, a, b))(lambdas)
    assert np.all(np.minimum(a, b) <= ys)
    assert np.all(ys <= np.maximum(a, b))


def test_handle_ring_opening_closing():
    rng = np.random.default_rng(2022)
    n = 100
    k = 20

    def random_ks():
        return rng.uniform(1, 2, (n,))

    src_k = random_ks()
    dst_k = random_ks()

    idxs = rng.choice(n, k, replace=False)

    closing_idxs = idxs[: k // 2]
    opening_idxs = idxs[k // 2 :]

    src_k[closing_idxs] = 0.0
    dst_k[opening_idxs] = 0.0

    lambda_min = 0.3
    lambda_max = 0.7

    f = functools.partial(
        handle_ring_opening_closing,
        linear_interpolation,
        src_k,
        dst_k,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
    )

    # closing
    # 0 < λ < λmin
    ks = f(0.5 * lambda_min)
    np.testing.assert_array_equal(ks[closing_idxs], 0.0)

    # λmax < λ < 1
    ks = f(0.5 * (1.0 + lambda_max))
    np.testing.assert_array_equal(ks[closing_idxs], dst_k[closing_idxs])

    # opening
    # 1 - λmin < λ < 1
    ks = f(0.5 * (2.0 - lambda_min))
    np.testing.assert_array_equal(ks[opening_idxs], 0.0)

    # 0 < λ < 1 - λmax
    ks = f(0.5 * (1.0 - lambda_max))
    np.testing.assert_array_equal(ks[opening_idxs], src_k[opening_idxs])


def test_interpolate_harmonic_force_constant():

    rng = np.random.default_rng(2022)

    def random_ks():
        return rng.uniform(1, 2, (100,))

    src_k = random_ks()
    dst_k = random_ks()

    k_min = 0.1
    assert (k_min < np.minimum(src_k, dst_k)).all()

    f = functools.partial(
        interpolate_harmonic_force_constant,
        k_min=k_min,
        lambda_min=0.0,
        lambda_max=0.4,
    )

    assert_interpolation_valid(f, src_k, dst_k)

    # check for sublinearity
    lam = 0.1
    k_linear = linear_interpolation(src_k, dst_k, lam)
    k_loglinear = f(src_k, dst_k, lam)
    assert np.all(k_loglinear < k_linear)

    # check for JIT-compatibility
    f = jax.jit(f)
    _ = f(src_k, dst_k, 0.1)


def test_cyclic_difference():
    assert cyclic_difference(0, 0, 1) == 0
    assert cyclic_difference(0, 1, 2) == 1  # arbitrary, positive by convention
    assert cyclic_difference(0, 0, 3) == 0
    assert cyclic_difference(0, 1, 3) == 1
    assert cyclic_difference(0, 2, 3) == -1

    _ = jax.jit(cyclic_difference)(0, 1, 1)


def assert_linear(f, x1, x2, x3):
    np.testing.assert_allclose(
        (f(x2) - f(x1)) / (x2 - x1),
        (f(x3) - f(x2)) / (x3 - x2),
    )
