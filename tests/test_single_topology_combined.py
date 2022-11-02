# test code for combining host guest systems and ensuring that parameters
# and lambda configurations are correct
import numpy as np
import pytest

from timemachine.constants import DEFAULT_FF
from timemachine.fe.interpolate import linear_interpolation
from timemachine.fe.single_topology import SingleTopology
from timemachine.fe.system import convert_bps_into_system
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.md import builders
from timemachine.testsystems.dhfr import get_dhfr_system
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


@pytest.fixture(scope="module")
def hif2a_ligand_pair_single_topology():
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file(DEFAULT_FF)
    return SingleTopology(mol_a, mol_b, core, forcefield)


def convert_omm_system(omm_system):
    bps, masses = openmm_deserializer.deserialize_system(omm_system, cutoff=1.2)
    num_atoms = len(masses)
    system = convert_bps_into_system(bps)
    return system, num_atoms


@pytest.fixture(scope="module")
def complex_host_system():
    host_sys_omm = get_dhfr_system()
    return convert_omm_system(host_sys_omm)


@pytest.fixture(scope="module")
def solvent_host_system():
    forcefield = Forcefield.load_from_file(DEFAULT_FF)
    host_sys_omm, _, _, _ = builders.build_water_system(3.0, forcefield.water_ff)
    return convert_omm_system(host_sys_omm)


@pytest.mark.parametrize("host_system_fixture", ["solvent_host_system", "complex_host_system"])
def test_combined_parameters_bonded(host_system_fixture, hif2a_ligand_pair_single_topology, request):
    # test bonded and nonbonded parameters are correct at the end-states.
    # 1) we expected bonded idxs in the ligand to be shifted by num_host_atoms
    # 2) we expected nonbonded lambda_idxs to be shifted
    # 3) we expected nonbonded parameters on the core to be linearly interpolated

    st = hif2a_ligand_pair_single_topology
    host_sys, num_host_atoms = request.getfixturevalue(host_system_fixture)

    def check_bonded_idxs_consistency(bonded_idxs, num_host_idxs):

        for b_idx, atom_idxs in enumerate(bonded_idxs):
            if b_idx < num_host_idxs:
                assert np.all(atom_idxs < num_host_atoms)
            else:
                assert np.all(atom_idxs >= num_host_atoms)

    for lamb in [0.0, 1.0]:

        # generate host guest system
        hgs = st.combine_with_host(host_sys, lamb=lamb)

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


@pytest.mark.parametrize("host_system_fixture", ["solvent_host_system", "complex_host_system"])
def test_combined_parameters_nonbonded(host_system_fixture, hif2a_ligand_pair_single_topology, request):
    # test bonded and nonbonded parameters are correct at the end-states.
    # 1) we expected bonded idxs in the ligand to be shifted by num_host_atoms
    # 2) we expected nonbonded lambda_idxs to be shifted
    # 3) we expected nonbonded parameters on the core to be linearly interpolated

    st = hif2a_ligand_pair_single_topology
    host_sys, num_host_atoms = request.getfixturevalue(host_system_fixture)

    for lamb in [0.0, 1.0]:

        hgs = st.combine_with_host(host_sys, lamb=lamb)
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

        for i in range(st.get_num_atoms()):
            for j in range(st.get_num_atoms()):
                if i < j:
                    expected_guest_idxs.add((i + num_host_atoms, j + num_host_atoms))

        assert hgs_exc_guest_idxs == expected_guest_idxs

        # 2) decoupling parameters for host-guest interactions
        # 2a) w offsets
        assert hgs.nonbonded_host_guest.params is not None
        w_coords = hgs.nonbonded_host_guest.params[:, 3]
        cutoff = hgs.nonbonded_host_guest.get_cutoff()

        for a_idx, w in enumerate(w_coords):
            if a_idx < num_host_atoms:
                # host atom
                assert w == 0.0
            else:
                # guest atom
                guest_atom_idx = a_idx - num_host_atoms
                indicator = st.c_flags[guest_atom_idx]
                if indicator == 0:
                    # core
                    assert w == 0.0
                elif indicator == 1:
                    # mol_a dummy
                    if lamb == 0.0:
                        assert w == 0.0
                    elif lamb == 1.0:
                        assert w == pytest.approx(cutoff)
                elif indicator == 2:
                    # mol_b dummy
                    if lamb == 0.0:
                        assert w == pytest.approx(cutoff)
                    elif lamb == 1.0:
                        assert w == 0.0
                else:
                    assert 0

        # 2b) nonbonded parameter interpolation checks
        mol_a_charges = st.ff.q_handle.parameterize(st.mol_a)
        mol_a_sig_eps = st.ff.lj_handle.parameterize(st.mol_a)

        mol_b_charges = st.ff.q_handle.parameterize(st.mol_b)
        mol_b_sig_eps = st.ff.lj_handle.parameterize(st.mol_b)

        for a_idx, (test_q, test_sig, test_eps, _) in enumerate(hgs.nonbonded_host_guest.params):

            if a_idx < num_host_atoms:
                continue

            guest_atom_idx = a_idx - num_host_atoms
            indicator = st.c_flags[guest_atom_idx]

            # dummy atom qlj parameters are arbitrary (since they will be decoupled via lambda parameters)
            if indicator != 0:
                continue

            if lamb == 0.0:
                # should resemble mol_a at lambda=0
                ref_q = mol_a_charges[st.c_to_a[guest_atom_idx]]
                ref_sig, ref_eps = mol_a_sig_eps[st.c_to_a[guest_atom_idx]]

                assert ref_q == test_q
                assert test_sig == ref_sig
                assert test_eps == ref_eps

            elif lamb == 1.0:
                # should resemble mol_b at lambda=1
                ref_q = mol_b_charges[st.c_to_b[guest_atom_idx]]
                ref_sig, ref_eps = mol_b_sig_eps[st.c_to_b[guest_atom_idx]]

                assert ref_q == test_q
                assert test_sig == ref_sig
                assert test_eps == ref_eps


@pytest.mark.parametrize("host_system_fixture", ["solvent_host_system", "complex_host_system"])
def test_combined_parameters_nonbonded_intermediate(
    host_system_fixture, hif2a_ligand_pair_single_topology: SingleTopology, request
):
    st = hif2a_ligand_pair_single_topology
    host_sys, num_host_atoms = request.getfixturevalue(host_system_fixture)

    rng = np.random.default_rng(2022)

    for lamb in rng.uniform(0.01, 0.99, (10,)):
        hgs = st.combine_with_host(host_sys, lamb=lamb)
        cutoff = hgs.nonbonded_host_guest.get_cutoff()

        assert hgs.nonbonded_host_guest.params is not None
        guest_params = hgs.nonbonded_host_guest.params[num_host_atoms:]
        ws_core = [w for flag, (_, _, _, w) in zip(st.c_flags, guest_params) if flag == 0]
        ws_a = [w for flag, (_, _, _, w) in zip(st.c_flags, guest_params) if flag == 1]
        ws_b = [w for flag, (_, _, _, w) in zip(st.c_flags, guest_params) if flag == 2]

        # core atoms fixed at w = 0
        assert all(w == 0.0 for w in ws_core)

        # dummy groups have consistent w coords
        assert len(np.unique(ws_a)) == 1
        assert len(np.unique(ws_b)) == 1
        w_a = np.unique(ws_a)[0]
        w_b = np.unique(ws_b)[0]

        # w in [0, cutoff]
        assert 0 < w_a < cutoff
        assert 0 < w_b < cutoff

        if lamb < 0.5:
            assert w_a < w_b
        else:
            assert w_b < w_a
