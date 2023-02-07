import functools
from importlib import resources

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from common import load_split_forcefields
from hypothesis import assume, given, seed
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine.constants import DEFAULT_FF
from timemachine.fe import atom_mapping, single_topology
from timemachine.fe.interpolate import linear_interpolation, log_linear_interpolation
from timemachine.fe.single_topology import (
    ChargePertubationError,
    CoreBondChangeWarning,
    MultipleAnchorWarning,
    SingleTopology,
    canonicalize_improper_idxs,
    cyclic_difference,
    handle_ring_opening_closing,
    interpolate_harmonic_force_constant,
    interpolate_w_coord,
    setup_dummy_interactions_from_ff,
)
from timemachine.fe.system import convert_bps_into_system, minimize_scipy, simulate_system
from timemachine.fe.utils import get_mol_name, get_romol_conf, read_sdf
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.md import minimizer
from timemachine.md.builders import build_water_system
from timemachine.potentials.jax_utils import pairwise_distances


def test_phenol():
    """
    Test that dummy interactions are setup correctly for a phenol. We want to check that bonds and angles
    are present when either a single root anchor is provided, or when a root anchor and a neighbor anchor is provided.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    AllChem.EmbedMolecule(mol, randomSeed=2022)

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

    with pytest.raises(single_topology.MissingAngleError):
        all_idxs, _ = setup_dummy_interactions_from_ff(ff, mol, dummy_group=[12], root_anchor_atom=6, nbr_anchor_atom=4)


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

    dgs, jks = single_topology.find_dummy_groups_and_anchors(mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1])
    assert dgs == [{3}]
    assert jks == [(2, 1)]

    # angle should swap
    core_pairs = np.array([[1, 2], [2, 0], [3, 1]])

    with pytest.warns(CoreBondChangeWarning):
        dgs, jks = single_topology.find_dummy_groups_and_anchors(mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1])
        assert dgs == [{3}]
        assert jks == [(2, None)]


def test_find_dummy_groups_and_anchors_multiple_angles():
    """
    Test that when multiple angle groups are possible we can find one deterministically
    """
    mol_a = Chem.MolFromSmiles("CCC")
    mol_b = Chem.MolFromSmiles("CC(C)C")

    AllChem.EmbedMolecule(mol_a, randomSeed=2022)
    AllChem.EmbedMolecule(mol_b, randomSeed=2022)

    core_pairs = np.array([[0, 2], [1, 1], [2, 3]])
    dgs, jks = single_topology.find_dummy_groups_and_anchors(mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1])
    assert dgs == [{0}]
    assert jks == [(1, 2)] or jks == [(1, 3)]

    dgs_zero, jks_zero = single_topology.find_dummy_groups_and_anchors(mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1])

    # this code should be invariant to different random seeds and different ordering of core pairs
    for idx in range(100):
        np.random.seed(idx)
        core_pairs_shuffle = np.random.permutation(core_pairs)
        dgs, jks = single_topology.find_dummy_groups_and_anchors(
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

    AllChem.EmbedMolecule(mol_a, randomSeed=2022)
    AllChem.EmbedMolecule(mol_b, randomSeed=2022)

    core_pairs = np.array([[1, 1], [2, 2]])

    with pytest.warns(MultipleAnchorWarning):
        dgs, jks = single_topology.find_dummy_groups_and_anchors(mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1])
        assert dgs == [{0}]
        assert jks == [(1, 2)] or jks == [(2, 1)]

    # test determinism, should be robust against seeds
    dgs_zero, jks_zero = single_topology.find_dummy_groups_and_anchors(mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1])
    for idx in range(100):
        np.random.seed(idx)
        core_pairs_shuffle = np.random.permutation(core_pairs)
        dgs, jks = single_topology.find_dummy_groups_and_anchors(
            mol_a, mol_b, core_pairs_shuffle[:, 0], core_pairs_shuffle[:, 1]
        )
        assert dgs == dgs_zero
        assert jks == jks_zero

    mol_a = Chem.MolFromSmiles("C(C)(C)C")
    mol_b = Chem.MolFromSmiles("O1CCCC1")

    AllChem.EmbedMolecule(mol_a, randomSeed=2022)
    AllChem.EmbedMolecule(mol_b, randomSeed=2022)

    core_a = [0, 1, 2, 3]
    core_b = [2, 1, 4, 3]

    with pytest.warns(MultipleAnchorWarning):
        dgs, jks = single_topology.find_dummy_groups_and_anchors(mol_a, mol_b, core_a, core_b)
        assert dgs == [{0}]
        assert jks == [(1, 2)]


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
        mols = read_sdf(path_to_ligand)

    pairs = [(mol_a, mol_b) for mol_a in mols for mol_b in mols]

    np.random.seed(2023)
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

    AllChem.EmbedMolecule(mol_a, randomSeed=2022)
    AllChem.EmbedMolecule(mol_b, randomSeed=2022)

    core = np.array([[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    st = SingleTopology(mol_a, mol_b, core, ff)

    test_masses = st.combine_masses()
    ref_masses = [Br_mass, C_mass, C_mass, max(C_mass, N_mass), C_mass, C_mass, C_mass, F_mass]
    np.testing.assert_almost_equal(test_masses, ref_masses)


@pytest.mark.nogpu
def test_jax_transform_intermediate_potential():
    def setup_arbitary_transformation():
        # NOTE: test system can probably be simplified; we just need
        # any SingleTopology and conformation
        with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
            mols = {get_mol_name(mol): mol for mol in read_sdf(path_to_ligand)}

        mol_a = mols["206"]
        mol_b = mols["57"]

        core = _get_core_by_mcs(mol_a, mol_b)
        ff = Forcefield.load_from_file(DEFAULT_FF)
        st = SingleTopology(mol_a, mol_b, core, ff)
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

    AllChem.EmbedMolecule(mol_a, randomSeed=2022)
    AllChem.EmbedMolecule(mol_b, randomSeed=2022)

    core = np.array([[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    solvent_sys, _, _, _ = build_water_system(4.0, ff.water_ff)
    host_bps, _ = openmm_deserializer.deserialize_system(solvent_sys, cutoff=1.2)

    st = SingleTopology(mol_a, mol_b, core, ff)
    host_system = st.combine_with_host(convert_bps_into_system(host_bps), 0.5)
    # Expect there to be 5 functions, excluding the chiral bond and chiral atom restraints
    # This should be updated when chiral restraints are re-enabled.
    assert len(host_system.get_U_fns()) == 5


@pytest.mark.parametrize("precision, rtol, atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
def test_nonbonded_split(precision, rtol, atol):
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols = {get_mol_name(mol): mol for mol in read_sdf(path_to_ligand)}
    mol_a = mols["338"]
    mol_b = mols["43"]
    core = _get_core_by_mcs(mol_a, mol_b)

    # split forcefield has different parameters for intramol and intermol terms
    ffs = load_split_forcefields()
    solvent_sys, solvent_conf, solvent_box, solvent_top = build_water_system(4.0, ffs.ref.water_ff)
    solvent_bps, _ = openmm_deserializer.deserialize_system(solvent_sys, cutoff=1.2)

    def get_vacuum_solvent_u_grads(ff, lamb):
        st = SingleTopology(mol_a, mol_b, core, ff)
        ligand_conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))  # TODO put lamb here w/Josh's fix
        combined_conf = np.concatenate([solvent_conf, ligand_conf])

        vacuum_system = st.setup_intermediate_state(lamb)
        vacuum_potentials = vacuum_system.get_U_fns()
        vacuum_impls = [p.bound_impl(precision) for p in vacuum_potentials]
        val_and_grad_fn = minimizer.get_val_and_grad_fn(vacuum_impls, solvent_box)
        vacuum_u, vacuum_grad = val_and_grad_fn(ligand_conf)

        solvent_system = st.combine_with_host(convert_bps_into_system(solvent_bps), lamb)
        solvent_potentials = solvent_system.get_U_fns()
        solvent_impls = [p.bound_impl(precision) for p in solvent_potentials]
        val_and_grad_fn = minimizer.get_val_and_grad_fn(solvent_impls, solvent_box)
        solvent_u, solvent_grad = val_and_grad_fn(combined_conf)
        return vacuum_grad, vacuum_u, solvent_grad, solvent_u

    n_lambdas = 3
    for lamb in np.linspace(0, 1, n_lambdas):
        # Compute the grads, potential with the ref ff
        vacuum_grad_ref, vacuum_u_ref, solvent_grad_ref, solvent_u_ref = get_vacuum_solvent_u_grads(ffs.ref, lamb)

        # Compute the grads, potential with the scaled ff
        vacuum_grad_scaled, vacuum_u_scaled, solvent_grad_scaled, solvent_u_scaled = get_vacuum_solvent_u_grads(
            ffs.scaled, lamb
        )

        # Compute the grads, potential with the intermol scaled ff
        (
            vacuum_grad_inter_scaled,
            vacuum_u_inter_scaled,
            solvent_grad_inter_scaled,
            solvent_u_inter_scaled,
        ) = get_vacuum_solvent_u_grads(ffs.inter_scaled, lamb)

        # Compute the expected intermol scaled potential
        expected_inter_scaled_u = solvent_u_scaled - vacuum_u_scaled + vacuum_u_ref

        # Pad gradients for the solvent
        vacuum_grad_scaled_padded = np.concatenate([np.zeros(solvent_conf.shape), vacuum_grad_scaled])
        vacuum_grad_ref_padded = np.concatenate([np.zeros(solvent_conf.shape), vacuum_grad_ref])
        expected_inter_scaled_grad = solvent_grad_scaled - vacuum_grad_scaled_padded + vacuum_grad_ref_padded

        # They should be equal
        assert expected_inter_scaled_u == pytest.approx(solvent_u_inter_scaled, rel=rtol, abs=atol)
        np.testing.assert_allclose(expected_inter_scaled_grad, solvent_grad_inter_scaled, rtol=rtol, atol=atol)

        # The vacuum term should be the same as the ref
        assert vacuum_u_inter_scaled == pytest.approx(vacuum_u_ref, rel=rtol, abs=atol)
        np.testing.assert_allclose(vacuum_grad_ref, vacuum_grad_inter_scaled, rtol=rtol, atol=atol)


def ligand_from_smiles(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.Compute2DCoords(mol)
    return mol


def _get_core_by_mcs(mol_a, mol_b):
    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        ring_cutoff=0.12,
        chain_cutoff=0.2,
        max_visits=1e7,
        connected_core=True,
        max_cores=1e6,
        enforce_core_core=True,
        complete_rings=True,
        enforce_chiral=True,
        min_threshold=0,
    )

    core = all_cores[0]
    return core


def test_no_chiral_atom_restraints():
    mol_a = ligand_from_smiles("c1ccccc1")
    mol_b = ligand_from_smiles("c1(I)ccccc1")
    core = _get_core_by_mcs(mol_a, mol_b)

    forcefield = Forcefield.load_from_file(DEFAULT_FF)
    st = SingleTopology(mol_a, mol_b, core, forcefield)
    init_conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))
    state = st.setup_intermediate_state(0.1)

    assert len(state.chiral_atom.get_idxs()) == 0
    U = state.get_U_fn()
    _ = U(init_conf)


def test_no_chiral_bond_restraints():
    mol_a = ligand_from_smiles("C")
    mol_b = ligand_from_smiles("CI")
    core = _get_core_by_mcs(mol_a, mol_b)

    forcefield = Forcefield.load_from_file(DEFAULT_FF)
    st = SingleTopology(mol_a, mol_b, core, forcefield)
    init_conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))
    state = st.setup_intermediate_state(0.1)

    assert len(state.chiral_bond.get_idxs()) == 0
    U = state.get_U_fn()
    _ = U(init_conf)


finite_floats = functools.partial(st.floats, allow_nan=False, allow_infinity=False, allow_subnormal=False)

nonzero_force_constants = finite_floats(1e-9, 1e9)

lambdas = finite_floats(0.0, 1.0)


def pairs(elem, unique=False):
    return st.lists(elem, min_size=2, max_size=2, unique=unique).map(tuple)


lambda_intervals = pairs(finite_floats(1e-9, 1.0 - 1e-9), unique=True).map(sorted)


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


@given(nonzero_force_constants, st.lists(lambdas, min_size=3, max_size=3, unique=True).map(sorted))
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


def test_interpolate_harmonic_force_constant_jax_transformable():
    _ = jax.jit(interpolate_harmonic_force_constant)(0.0, 1.0, 0.1, 1e-12, 0.0, 1.0)


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


@given(bounded_ints, bounded_ints, periods)
@seed(2022)
def test_cyclic_difference_inverse(a, b, period):
    x = cyclic_difference(a, b, period)
    assert np.abs(x) <= period / 2
    assert_equal_cyclic(a + x, b, period)


@given(bounded_ints, bounded_ints, periods)
@seed(2022)
def test_cyclic_difference_antisymmetric(a, b, period):
    assert cyclic_difference(a, b, period) + cyclic_difference(b, a, period) == 0


@given(bounded_ints, bounded_ints, bounded_ints, bounded_ints, periods)
@seed(2022)
def test_cyclic_difference_shift_by_n_periods(a, b, m, n, period):
    assert_equal_cyclic(
        cyclic_difference(a + m * period, b + n * period, period),
        cyclic_difference(a, b, period),
        period,
    )


@given(bounded_ints, bounded_ints, bounded_ints, periods)
@seed(2022)
def test_cyclic_difference_translation_invariant(a, b, t, period):
    assert_equal_cyclic(
        cyclic_difference(a + t, b + t, period),
        cyclic_difference(a, b, period),
        period,
    )


@given(pairs(finite_floats()))
@seed(2022)
def test_interpolate_w_coord_valid_at_end_states(end_states):
    a, b = end_states
    f = functools.partial(interpolate_w_coord, a, b)
    assert f(0.0) == a
    assert f(1.0) == b


def test_interpolate_w_coord_monotonic():
    lambdas = np.linspace(0.0, 1.0, 100)
    ws = interpolate_w_coord(0.0, 1.0, lambdas)
    assert np.all(np.diff(ws) >= 0.0)
