# test code for combining host guest systems and ensuring that parameters
# and lambda configurations are correct
import numpy as np

from timemachine.fe.system import convert_bps_into_system
from timemachine.ff.handlers import openmm_deserializer
from timemachine.md import builders
from timemachine.testsystems.dhfr import get_dhfr_system
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


def test_combined_parameters_solvent():
    host_sys_omm, _, _, _ = builders.build_water_system(3.0)
    _test_combined_parameters_impl_bonded(host_sys_omm)
    _test_combined_parameters_impl_nonbonded(host_sys_omm)


def test_combined_parameters_complex():
    host_sys_omm = get_dhfr_system()
    _test_combined_parameters_impl_bonded(host_sys_omm)
    _test_combined_parameters_impl_nonbonded(host_sys_omm)


def _test_combined_parameters_impl_bonded(host_system_omm):
    # test bonded and nonbonded parameters are correct at the end-states.
    # 1) we expected bonded idxs in the ligand to be shifted by num_host_atoms
    # 2) we expected nonbonded lambda_idxs to be shifted
    # 3) we expected nonbonded parameters on the core to be linearly interpolated

    st3 = get_hif2a_ligand_pair_single_topology()
    host_bps, masses = openmm_deserializer.deserialize_system(host_system_omm, cutoff=1.2)
    num_host_atoms = len(masses)
    host_sys = convert_bps_into_system(host_bps)

    def check_bonded_idxs_consistency(bonded_idxs, num_host_idxs):

        for b_idx, atom_idxs in enumerate(bonded_idxs):
            if b_idx < num_host_idxs:
                assert np.all(atom_idxs < num_host_atoms)
            else:
                assert np.all(atom_idxs >= num_host_atoms)

    for lamb in [0.0, 1.0]:

        # generate host guest system
        hgs = st3.combine_with_host(host_sys, lamb=lamb)

        # check bonds
        check_bonded_idxs_consistency(hgs.bond.get_idxs(), len(host_sys.bond.get_idxs()))
        check_bonded_idxs_consistency(hgs.angle.get_idxs(), len(host_sys.angle.get_idxs()))

        if host_sys.torsion:
            check_bonded_idxs_consistency(hgs.torsion.get_idxs(), len(host_sys.torsion.get_idxs()))
        else:
            check_bonded_idxs_consistency(hgs.torsion.get_idxs(), 0)

        if host_sys.chiral_atom:
            check_bonded_idxs_consistency(hgs.chiral_atom.get_idxs(), len(host_sys.chiral_atom.get_idxs()))
        else:
            check_bonded_idxs_consistency(hgs.chiral_atom.get_idxs(), 0)

        if host_sys.chiral_bond:
            check_bonded_idxs_consistency(hgs.chiral_bond.get_idxs(), len(host_sys.chiral_bond.get_idxs()))
        else:
            check_bonded_idxs_consistency(hgs.chiral_bond.get_idxs(), 0)

        check_bonded_idxs_consistency(hgs.nonbonded_guest_pairs.get_idxs(), 0)


def _test_combined_parameters_impl_nonbonded(host_system_omm):
    # test bonded and nonbonded parameters are correct at the end-states.
    # 1) we expected bonded idxs in the ligand to be shifted by num_host_atoms
    # 2) we expected nonbonded lambda_idxs to be shifted
    # 3) we expected nonbonded parameters on the core to be linearly interpolated

    st3 = get_hif2a_ligand_pair_single_topology()
    host_bps, masses = openmm_deserializer.deserialize_system(host_system_omm, cutoff=1.2)
    num_host_atoms = len(masses)
    host_sys = convert_bps_into_system(host_bps)

    for lamb in [0.0, 1.0]:

        hgs = st3.combine_with_host(host_sys, lamb=lamb)
        # check nonbonded terms
        # 1) exclusions
        # exclusions should be set for all ligand ixns in hgs.nonbonded_host_guest
        hgs_exc_idxs = hgs.nonbonded_host_guest.get_exclusion_idxs()
        hgs_exc_guest_idxs = set()
        for i, j in hgs_exc_idxs:
            if i > num_host_atoms or j > num_host_atoms:
                assert (i, j) not in hgs_exc_guest_idxs
                hgs_exc_guest_idxs.add((i, j))

        expected_guest_idxs = set()

        for i in range(st3.get_num_atoms()):
            for j in range(st3.get_num_atoms()):
                if i < j:
                    expected_guest_idxs.add((i + num_host_atoms, j + num_host_atoms))

        assert hgs_exc_guest_idxs == expected_guest_idxs

        # 2) decoupling parameters for host-guest interactions
        # 2a) lambda offset and plane parameters
        hgs_lambda_offset_idxs = hgs.nonbonded_host_guest.get_lambda_offset_idxs()
        hgs_lambda_plane_idxs = hgs.nonbonded_host_guest.get_lambda_plane_idxs()

        for a_idx, (offset_idx, plane_idx) in enumerate(zip(hgs_lambda_offset_idxs, hgs_lambda_plane_idxs)):
            if a_idx < num_host_atoms:
                assert offset_idx == 0
                assert plane_idx == 0
            else:
                # guest atom
                guest_atom_idx = a_idx - num_host_atoms
                indicator = st3.c_flags[guest_atom_idx]
                if indicator == 0:
                    # core
                    assert offset_idx == 0
                    assert plane_idx == 0
                elif indicator == 1:
                    # mol_a dummy
                    assert offset_idx == 1
                    assert plane_idx == 0
                elif indicator == 2:
                    # mol_b dummy
                    assert offset_idx == 1
                    assert plane_idx == -1
                else:
                    assert 0

        # 2b) nonbonded parameter interpolation checks
        mol_a_charges = st3.ff.q_handle.parameterize(st3.mol_a)
        mol_a_sig_eps = st3.ff.lj_handle.parameterize(st3.mol_a)

        mol_b_charges = st3.ff.q_handle.parameterize(st3.mol_b)
        mol_b_sig_eps = st3.ff.lj_handle.parameterize(st3.mol_b)

        for a_idx, (test_q, test_sig, test_eps) in enumerate(hgs.nonbonded_host_guest.params):

            if a_idx < num_host_atoms:
                continue

            guest_atom_idx = a_idx - num_host_atoms
            indicator = st3.c_flags[guest_atom_idx]

            # dummy atom qlj parameters are arbitrary (since they will be decoupled via lambda parameters)
            if indicator != 0:
                continue

            if lamb == 0.0:
                # should resemble mol_a at lambda=0
                ref_q = mol_a_charges[st3.c_to_a[guest_atom_idx]]
                ref_sig, ref_eps = mol_a_sig_eps[st3.c_to_a[guest_atom_idx]]

                assert ref_q == test_q
                assert test_sig == ref_sig
                assert test_eps == ref_eps

            elif lamb == 1.0:
                # should resemble mol_b at lambda=1
                ref_q = mol_b_charges[st3.c_to_b[guest_atom_idx]]
                ref_sig, ref_eps = mol_b_sig_eps[st3.c_to_b[guest_atom_idx]]

                assert ref_q == test_q
                assert test_sig == ref_sig
                assert test_eps == ref_eps
