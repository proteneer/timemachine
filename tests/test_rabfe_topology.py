# test topology classes used in the RABFE protocol.
from jax.config import config

config.update("jax_enable_x64", True)
from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers

from fe import topology
from rdkit import Chem
import numpy as np

from timemachine.lib import potentials


def test_base_topology_conversion_ring_torsion():

    # test that the conversion protocol behaves as intended on a
    # simple linked cycle.

    ff_handlers = deserialize_handlers(open("ff/params/smirnoff_1_1_0_sc.py").read())
    ff = Forcefield(ff_handlers)
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
    np.testing.assert_array_equal(topology.standard_qlj_typer(mol), dst_qlj_params)


def test_base_topology_conversion_r_group():

    # check that phenol torsions are turned off
    ff_handlers = deserialize_handlers(open("ff/params/smirnoff_1_1_0_sc.py").read())
    ff = Forcefield(ff_handlers)
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    mol_top = topology.BaseTopologyConversion(mol, ff)
    result, potential = mol_top.parameterize_proper_torsion(ff.pt_handle.params)
    # in the conversion phase, torsions that bridge the two rings should be set to
    # be alchemically turned off.
    assert potential.get_lambda_mult() is None
    assert potential.get_lambda_offset() is None


def test_base_topology_standard_decoupling():

    # this class is typically used in the second step of the RABFE protocol for the solvent leg.
    # we expected the charges to be zero, and the lj parameters to be standardized. In addition,
    # the torsions should be turned off.
    ff_handlers = deserialize_handlers(open("ff/params/smirnoff_1_1_0_sc.py").read())
    ff = Forcefield(ff_handlers)
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    vanilla_mol_top = topology.BaseTopology(mol, ff)
    vanilla_torsion_params, _ = vanilla_mol_top.parameterize_proper_torsion(ff.pt_handle.params)

    mol_top = topology.BaseTopologyStandardDecoupling(mol, ff)
    decouple_torsion_params, torsion_potential = mol_top.parameterize_proper_torsion(ff.pt_handle.params)

    np.testing.assert_array_equal(vanilla_torsion_params, decouple_torsion_params)

    # in the conversion phase, torsions that bridge the two rings should be set to
    # be alchemically turned off.
    is_in_ring = [1, 1, 1, 1, 1, 1, 0, 0]

    combined_decouple_torsion_params, combined_torsion_potential = mol_top.parameterize_periodic_torsion(
        ff.pt_handle.params, ff.it_handle.params
    )

    assert len(combined_torsion_potential.get_lambda_mult()) == len(combined_torsion_potential.get_idxs())
    assert len(combined_torsion_potential.get_lambda_mult()) == len(combined_torsion_potential.get_lambda_offset())

    # impropers should always be turned on.
    num_proper_torsions = len(torsion_potential.get_idxs())

    assert np.all(combined_torsion_potential.get_lambda_mult() == 0)
    assert np.all(combined_torsion_potential.get_lambda_offset() == 1)

    qlj_params, nonbonded_potential = mol_top.parameterize_nonbonded(ff.q_handle.params, ff.lj_handle.params)

    assert not isinstance(nonbonded_potential, potentials.NonbondedInterpolated)

    np.testing.assert_array_equal(topology.standard_qlj_typer(mol), qlj_params)

    np.testing.assert_array_equal(
        nonbonded_potential.get_lambda_plane_idxs(), np.zeros(mol.GetNumAtoms(), dtype=np.int32)
    )
    np.testing.assert_array_equal(
        nonbonded_potential.get_lambda_offset_idxs(), np.ones(mol.GetNumAtoms(), dtype=np.int32)
    )


def test_dual_topology_standard_decoupling():

    # this class is used in double decoupling stages of the RABFE protocol. It modifies the
    # DualTopology class in two ways:
    # 1) the torsions between non-ring atoms are turned off
    # 2) the nonbonded terms are standardized, but also interpolated at lambda=0 such that
    #    the epsilons are at half strength.

    ff_handlers = deserialize_handlers(open("ff/params/smirnoff_1_1_0_sc.py").read())
    ff = Forcefield(ff_handlers)
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
    num_proper_torsions = len(torsion_potential.get_idxs())

    assert np.all(combined_torsion_potential.get_lambda_mult() == 0)
    assert np.all(combined_torsion_potential.get_lambda_offset() == 1)

    qlj_params, nonbonded_potential = mol_top.parameterize_nonbonded(ff.q_handle.params, ff.lj_handle.params)

    assert isinstance(nonbonded_potential, potentials.NonbondedInterpolated)

    expected_qlj = topology.standard_qlj_typer(mol_c)
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


def test_dual_topology_minimization():

    # Identical to the vanilla Dual Topology class, except that both ligands are
    # decouple simultaneously

    ff_handlers = deserialize_handlers(open("ff/params/smirnoff_1_1_0_sc.py").read())
    ff = Forcefield(ff_handlers)
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

    ff_handlers = deserialize_handlers(open("ff/params/smirnoff_1_1_0_sc.py").read())
    ff = Forcefield(ff_handlers)
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
