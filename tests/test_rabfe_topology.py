# test topology classes used in the RABFE protocol.
from jax.config import config

config.update("jax_enable_x64", True)
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine.constants import ONE_4PI_EPS0
from timemachine.fe import topology
from timemachine.ff import Forcefield
from timemachine.lib import potentials


def test_base_topology_conversion_ring_torsion():

    # test that the conversion protocol behaves as intended on a
    # simple linked cycle.

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    mol = Chem.MolFromSmiles("C1CC1C1CC1")
    vanilla_mol_top = topology.BaseTopology(mol, ff)
    vanilla_torsion_params, _ = vanilla_mol_top.parameterize_proper_torsion(ff.pt_handle.params)

    mol_top = topology.BaseTopologyConversion(mol, ff)
    conversion_torsion_params, torsion_potential = mol_top.parameterize_proper_torsion(ff.pt_handle.params)

    np.testing.assert_array_equal(vanilla_torsion_params, conversion_torsion_params)

    assert torsion_potential.get_lambda_mult() is None
    assert torsion_potential.get_lambda_offset() is None

    vanilla_qlj_params, _ = vanilla_mol_top.parameterize_nonbonded(ff.q_handle.params, ff.lj_handle.params)
    qlj_params, nonbonded_potential = mol_top.parameterize_nonbonded(ff.q_handle.params, ff.lj_handle.params)

    assert isinstance(nonbonded_potential, potentials.NonbondedInterpolated)

    src_qlj_params = qlj_params[: len(qlj_params) // 2]
    dst_qlj_params = qlj_params[len(qlj_params) // 2 :]

    np.testing.assert_array_equal(vanilla_qlj_params, src_qlj_params)

    expected_qlj_params = np.array(vanilla_qlj_params)
    expected_qlj_params[:, 0] = 0
    np.testing.assert_array_equal(expected_qlj_params, dst_qlj_params)


def test_base_topology_conversion_r_group():

    # check that phenol torsions are turned off
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    mol_top = topology.BaseTopologyConversion(mol, ff)
    result, potential = mol_top.parameterize_proper_torsion(ff.pt_handle.params)
    # in the conversion phase, torsions that bridge the two rings should be set to
    # be alchemically turned off.
    assert potential.get_lambda_mult() is None
    assert potential.get_lambda_offset() is None


def test_base_topology_standard_decoupling():

    # this class is typically used in the second step of the RABFE protocol for the solvent leg.
    # we expected the charges to be zero. In addition,
    # the torsions should be turned off.
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    vanilla_mol_top = topology.BaseTopology(mol, ff)
    vanilla_torsion_params, _ = vanilla_mol_top.parameterize_proper_torsion(ff.pt_handle.params)

    mol_top = topology.BaseTopologyStandardDecoupling(mol, ff)
    decouple_torsion_params, torsion_potential = mol_top.parameterize_proper_torsion(ff.pt_handle.params)

    np.testing.assert_array_equal(vanilla_torsion_params, decouple_torsion_params)

    # in the conversion phase, torsions that bridge the two rings should be set to
    # be alchemically turned off.
    # is_in_ring = [1, 1, 1, 1, 1, 1, 0, 0]

    combined_decouple_torsion_params, combined_torsion_potential = mol_top.parameterize_periodic_torsion(
        ff.pt_handle.params, ff.it_handle.params
    )

    assert len(combined_torsion_potential.get_lambda_mult()) == len(combined_torsion_potential.get_idxs())
    assert len(combined_torsion_potential.get_lambda_mult()) == len(combined_torsion_potential.get_lambda_offset())

    # impropers should always be turned on.
    assert np.all(combined_torsion_potential.get_lambda_mult() == 0)
    assert np.all(combined_torsion_potential.get_lambda_offset() == 1)

    qlj_params, nonbonded_potential = mol_top.parameterize_nonbonded(ff.q_handle.params, ff.lj_handle.params)

    assert not isinstance(nonbonded_potential, potentials.NonbondedInterpolated)

    np.testing.assert_array_equal(qlj_params[:, 0], np.zeros_like(qlj_params[:, 0]))

    np.testing.assert_array_equal(
        nonbonded_potential.get_lambda_plane_idxs(), np.zeros(mol.GetNumAtoms(), dtype=np.int32)
    )
    np.testing.assert_array_equal(
        nonbonded_potential.get_lambda_offset_idxs(), np.ones(mol.GetNumAtoms(), dtype=np.int32)
    )


def test_dual_topology_standard_decoupling():

    # this class is used in double decoupling stages of the RABFE protocol. It modifies the
    # DualTopology class in one way:
    # 1) the nonbonded terms are interpolated at lambda=0 such that the epsilons and charges are at half strength.

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    mol_a = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1F"))
    mol_c = Chem.CombineMols(mol_a, mol_b)
    mol_top = topology.DualTopologyStandardDecoupling(mol_a, mol_b, ff)

    decouple_torsion_params, torsion_potential = mol_top.parameterize_proper_torsion(ff.pt_handle.params)

    combined_decouple_torsion_params, combined_torsion_potential = mol_top.parameterize_periodic_torsion(
        ff.pt_handle.params, ff.it_handle.params
    )

    assert len(combined_torsion_potential.get_lambda_mult()) == len(combined_torsion_potential.get_idxs())
    assert len(combined_torsion_potential.get_lambda_mult()) == len(combined_torsion_potential.get_lambda_offset())

    # impropers should always be turned on.
    # num_proper_torsions = len(torsion_potential.get_idxs())

    assert np.all(combined_torsion_potential.get_lambda_mult() == 0)
    assert np.all(combined_torsion_potential.get_lambda_offset() == 1)

    qlj_params, nonbonded_potential = mol_top.parameterize_nonbonded(ff.q_handle.params, ff.lj_handle.params)

    assert isinstance(nonbonded_potential, potentials.NonbondedInterpolated)

    expected_qlj = topology.standard_qlj_typer(mol_c)
    expected_qlj[:, 0] = expected_qlj[:, 0] / 2  # charges should be halved
    expected_qlj[:, 2] = expected_qlj[:, 2] / 2  # eps should be halved

    src_qlj_params = qlj_params[: len(qlj_params) // 2]
    dst_qlj_params = qlj_params[len(qlj_params) // 2 :]

    np.testing.assert_array_equal(src_qlj_params, expected_qlj)

    expected_qlj = topology.standard_qlj_typer(mol_c)

    np.testing.assert_array_equal(dst_qlj_params, expected_qlj)

    combined_lambda_plane_idxs = nonbonded_potential.get_lambda_plane_idxs()
    combined_lambda_offset_idxs = nonbonded_potential.get_lambda_offset_idxs()

    A = mol_a.GetNumAtoms()
    B = mol_b.GetNumAtoms()
    C = mol_c.GetNumAtoms()

    np.testing.assert_array_equal(combined_lambda_plane_idxs, np.zeros(C))
    np.testing.assert_array_equal(combined_lambda_offset_idxs[:A], np.zeros(A))
    np.testing.assert_array_equal(combined_lambda_offset_idxs[A:], np.ones(B))


def test_dual_topology_standard_decoupling_charged():

    # special test case for charged molecules, we expect the charges to be rescaled
    # based on each individual molecule's charge, as opposed to based on the sum
    # of the charges.

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    mol_a = Chem.AddHs(Chem.MolFromSmiles("C1CC1[O-]"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("C1[O+]CCCCC1"))

    mol_top = topology.DualTopologyStandardDecoupling(mol_a, mol_b, ff)

    qlj_params, nonbonded_potential = mol_top.parameterize_nonbonded(ff.q_handle.params, ff.lj_handle.params)

    assert isinstance(nonbonded_potential, potentials.NonbondedInterpolated)

    expected_qlj = np.concatenate([topology.standard_qlj_typer(mol_a), topology.standard_qlj_typer(mol_b)])

    # need to set the charges correctly, and manually
    N_A = mol_a.GetNumAtoms()
    N_B = mol_b.GetNumAtoms()

    expected_qlj[:N_A, 0] = -1.0 / N_A
    expected_qlj[N_A:, 0] = 1.0 / N_B
    expected_qlj[:, 2] = expected_qlj[:, 2]  # eps should be halved

    src_qlj_params = qlj_params[: len(qlj_params) // 2]
    dst_qlj_params = qlj_params[len(qlj_params) // 2 :]

    np.testing.assert_array_equal(dst_qlj_params, expected_qlj)

    expected_qlj[:N_A, 0] /= 2
    expected_qlj[N_A:, 0] /= 2
    expected_qlj[:, 2] /= 2  # eps should be halved

    np.testing.assert_array_equal(src_qlj_params[:, 0], expected_qlj[:, 0])


def test_dual_topology_minimization():

    # Identical to the vanilla Dual Topology class, except that both ligands are
    # decouple simultaneously

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    mol_a = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1F"))
    mol_top = topology.DualTopologyMinimization(mol_a, mol_b, ff)

    C = mol_a.GetNumAtoms() + mol_b.GetNumAtoms()

    _, nonbonded_potential = mol_top.parameterize_nonbonded(ff.q_handle.params, ff.lj_handle.params)

    assert not isinstance(nonbonded_potential, potentials.NonbondedInterpolated)

    np.testing.assert_array_equal(nonbonded_potential.get_lambda_offset_idxs(), np.ones(C, dtype=np.int32))
    np.testing.assert_array_equal(nonbonded_potential.get_lambda_plane_idxs(), np.zeros(C, dtype=np.int32))


def test_dual_topology_rhfe():

    # used in testing the relative hydration protocol. The nonbonded charges and epsilons are reduced
    # to half strength

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    mol_a = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1F"))
    mol_c = Chem.CombineMols(mol_a, mol_b)
    mol_top = topology.DualTopologyRHFE(mol_a, mol_b, ff)

    C = mol_a.GetNumAtoms() + mol_b.GetNumAtoms()

    ref_qlj_params, _ = topology.BaseTopology(mol_c, ff).parameterize_nonbonded(ff.q_handle.params, ff.lj_handle.params)

    qlj_params, nonbonded_potential = mol_top.parameterize_nonbonded(ff.q_handle.params, ff.lj_handle.params)

    assert isinstance(nonbonded_potential, potentials.NonbondedInterpolated)

    src_qlj_params = qlj_params[: len(qlj_params) // 2]
    dst_qlj_params = qlj_params[len(qlj_params) // 2 :]

    np.testing.assert_array_equal(src_qlj_params[:, 0], ref_qlj_params[:, 0] / 2)
    np.testing.assert_array_equal(src_qlj_params[:, 1], ref_qlj_params[:, 1])
    np.testing.assert_array_equal(src_qlj_params[:, 2], ref_qlj_params[:, 2] / 2)
    np.testing.assert_array_equal(dst_qlj_params, ref_qlj_params)

    combined_lambda_plane_idxs = nonbonded_potential.get_lambda_plane_idxs()
    combined_lambda_offset_idxs = nonbonded_potential.get_lambda_offset_idxs()

    A = mol_a.GetNumAtoms()
    B = mol_b.GetNumAtoms()
    C = mol_c.GetNumAtoms()

    np.testing.assert_array_equal(combined_lambda_plane_idxs, np.zeros(C))
    np.testing.assert_array_equal(combined_lambda_offset_idxs[:A], np.zeros(A))
    np.testing.assert_array_equal(combined_lambda_offset_idxs[A:], np.ones(B))


def test_dual_topology_charge_conversion():

    # test charge parameter interpolation for a charged molecule
    ff = Forcefield.load_from_file("smirnoff_1_1_0_ccc.py")

    #                                      12  3 45  67 89 0 1  2    3
    mol_a = Chem.AddHs(Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C([O-])=O"))  # 20 atoms (inc. Hs)
    mol_b = Chem.AddHs(Chem.MolFromSmiles("CC(-[O-])OC1=CC=CC=C1C(O)=O"))  # 22 atoms

    AllChem.EmbedMolecule(mol_a)
    AllChem.EmbedMolecule(mol_b)

    num_a_atoms = mol_a.GetNumAtoms()

    mol_top = topology.DualTopologyChargeConversion(mol_a, mol_b, ff)

    qlj_params, nonbonded_potential = mol_top.parameterize_nonbonded(ff.q_handle.params, ff.lj_handle.params)

    qlj_params_src = qlj_params[: len(qlj_params) // 2]
    qlj_params_dst = qlj_params[len(qlj_params) // 2 :]

    q_params_src = qlj_params_src[:, 0]
    q_params_dst = qlj_params_dst[:, 0]

    q_params_src_a = q_params_src[:num_a_atoms]
    q_params_src_b = q_params_src[num_a_atoms:]

    np.testing.assert_array_equal(q_params_src_b, np.zeros_like(q_params_src_b))
    np.testing.assert_almost_equal(np.sum(q_params_src_a), -1 * np.sqrt(ONE_4PI_EPS0), decimal=5)

    q_params_dst_a = q_params_dst[:num_a_atoms]
    q_params_dst_b = q_params_dst[num_a_atoms:]

    np.testing.assert_almost_equal(np.sum(q_params_dst_b), -1 * np.sqrt(ONE_4PI_EPS0), decimal=5)
    np.testing.assert_array_equal(q_params_dst_a, np.zeros_like(q_params_dst_a))

    # test that net charge along path of interpolation is -1
    for lam in np.linspace(0, 1, 100):
        qi = (1 - lam) * q_params_src + lam * q_params_dst
        np.testing.assert_almost_equal(np.sum(qi), -1 * np.sqrt(ONE_4PI_EPS0), decimal=5)
