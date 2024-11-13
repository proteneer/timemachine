# test code for combining host guest systems and ensuring that parameters
# and lambda configurations are correct
import jax
import numpy as np
import pytest

from timemachine import potentials
from timemachine.fe.single_topology import SingleTopology
from timemachine.fe.system import convert_omm_system
from timemachine.ff import Forcefield
from timemachine.md import builders
from timemachine.testsystems.dhfr import get_dhfr_system
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology

pytestmark = [pytest.mark.nocuda]


@pytest.fixture(scope="module")
def hif2a_ligand_pair_single_topology():
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_default()
    return SingleTopology(mol_a, mol_b, core, forcefield)


@pytest.fixture(scope="module")
def complex_host_system():
    host_sys_omm, host_top = get_dhfr_system()
    # Hardcoded to match 5dfr_solv_equil.pdb file
    num_water_atoms = 21069
    return convert_omm_system(host_sys_omm), num_water_atoms, host_top


@pytest.fixture(scope="module")
def solvent_host_system():
    ff = Forcefield.load_default()
    host_sys_omm, conf, _, top = builders.build_water_system(3.0, ff.water_ff)
    return convert_omm_system(host_sys_omm), conf.shape[0], top


@pytest.mark.parametrize("lamb", [0.0, 1.0])
@pytest.mark.parametrize("host_system_fixture", ["solvent_host_system", "complex_host_system"])
def test_combined_parameters_bonded(host_system_fixture, lamb, hif2a_ligand_pair_single_topology, request):
    # test bonded and nonbonded parameters are correct at the end-states.
    # 1) we expected bonded idxs in the ligand to be shifted by num_host_atoms
    # 2) we expected nonbonded lambda_idxs to be shifted
    # 3) we expected nonbonded parameters on the core to be linearly interpolated

    st = hif2a_ligand_pair_single_topology
    (host_sys, host_masses), num_water_atoms, omm_topology = request.getfixturevalue(host_system_fixture)
    num_host_atoms = len(host_masses)

    def check_bonded_idxs_consistency(bonded_idxs, num_host_idxs):
        for b_idx, atom_idxs in enumerate(bonded_idxs):
            if b_idx < num_host_idxs:
                assert np.all(atom_idxs < num_host_atoms)
            else:
                assert np.all(atom_idxs >= num_host_atoms)

    # generate host guest system
    hgs = st.combine_with_host(host_sys, lamb, num_water_atoms, st.ff, omm_topology)

    # check bonds
    check_bonded_idxs_consistency(hgs.bond.potential.idxs, len(host_sys.bond.potential.idxs))
    check_bonded_idxs_consistency(hgs.angle.potential.idxs, len(host_sys.angle.potential.idxs))
    check_bonded_idxs_consistency(hgs.proper.potential.idxs, len(host_sys.proper.potential.idxs))
    check_bonded_idxs_consistency(hgs.improper.potential.idxs, len(host_sys.improper.potential.idxs))

    if host_sys.chiral_atom:
        check_bonded_idxs_consistency(hgs.chiral_atom.potential.idxs, len(host_sys.chiral_atom.potential.idxs))
    else:
        check_bonded_idxs_consistency(hgs.chiral_atom.potential.idxs, 0)

    if host_sys.chiral_bond:
        check_bonded_idxs_consistency(hgs.chiral_bond.potential.idxs, len(host_sys.chiral_bond.potential.idxs))
    else:
        check_bonded_idxs_consistency(hgs.chiral_bond.potential.idxs, 0)

    check_bonded_idxs_consistency(hgs.nonbonded_guest_pairs.potential.idxs, 0)


@pytest.mark.parametrize("lamb", [0.0, 1.0])
@pytest.mark.parametrize("host_system_fixture", ["solvent_host_system", "complex_host_system"])
def test_combined_parameters_nonbonded(host_system_fixture, lamb, hif2a_ligand_pair_single_topology, request):
    # test bonded and nonbonded parameters are correct at the end-states.
    # 1) we expected bonded idxs in the ligand to be shifted by num_host_atoms
    # 2) we expected nonbonded lambda_idxs to be shifted
    # 3) we expected nonbonded parameters on the core to be linearly interpolated

    st = hif2a_ligand_pair_single_topology
    (host_sys, host_masses), num_water_atoms, omm_topology = request.getfixturevalue(host_system_fixture)
    num_host_atoms = len(host_masses)

    hgs = st.combine_with_host(host_sys, lamb, num_water_atoms, st.ff, omm_topology)
    # check nonbonded terms
    # 1) ligand ixns should be omitted in hgs.nonbonded_host
    assert isinstance(hgs.nonbonded_host.potential, potentials.Nonbonded)
    assert hgs.nonbonded_host.potential.atom_idxs is not None
    assert set(hgs.nonbonded_host.potential.atom_idxs) == set(range(num_host_atoms))

    # 2) decoupling parameters for host-guest interactions
    # 2a) w offsets
    potential = hgs.nonbonded_host_guest_ixn.potential
    params = hgs.nonbonded_host_guest_ixn.params

    # NBIxnGroup has the ligand interaction parameters
    assert isinstance(potential, potentials.NonbondedInteractionGroup)
    w_coords = params[:, 3]

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
                    assert w == potential.cutoff
            elif indicator == 2:
                # mol_b dummy
                if lamb == 0.0:
                    assert w == potential.cutoff
                elif lamb == 1.0:
                    assert w == 0.0
            else:
                assert 0

    # 2b) nonbonded parameter interpolation checks
    mol_a_charges = st.ff.q_handle.parameterize(st.mol_a)
    mol_a_sig_eps = st.ff.lj_handle.parameterize(st.mol_a)

    mol_b_charges = st.ff.q_handle.parameterize(st.mol_b)
    mol_b_sig_eps = st.ff.lj_handle.parameterize(st.mol_b)

    for a_idx, (test_q, test_sig, test_eps, _) in enumerate(params):
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


@pytest.mark.parametrize("lamb", np.random.default_rng(2022).uniform(0.01, 0.99, (10,)))
@pytest.mark.parametrize("host_system_fixture", ["solvent_host_system", "complex_host_system"])
def test_combined_parameters_nonbonded_intermediate(
    host_system_fixture, lamb, hif2a_ligand_pair_single_topology: SingleTopology, request
):
    st = hif2a_ligand_pair_single_topology
    (host_sys, host_masses), num_water_atoms, omm_topology = request.getfixturevalue(host_system_fixture)
    num_host_atoms = len(host_masses)

    hgs = st.combine_with_host(host_sys, lamb, num_water_atoms, st.ff, omm_topology)

    potential = hgs.nonbonded_host_guest_ixn.potential
    params = hgs.nonbonded_host_guest_ixn.params
    assert isinstance(potential, potentials.NonbondedInteractionGroup)

    guest_params = np.array(params[num_host_atoms:])
    ws_core = [w for flag, (_, _, _, w) in zip(st.c_flags, guest_params) if flag == 0]
    ws_a = [w for flag, (_, _, _, w) in zip(st.c_flags, guest_params) if flag == 1]
    ws_b = [w for flag, (_, _, _, w) in zip(st.c_flags, guest_params) if flag == 2]

    # core atoms fixed at w = 0
    assert all(w == 0.0 for w in ws_core)

    # dummy groups have consistent w coords
    assert len(set(ws_a)) == 1
    assert len(set(ws_b)) == 1
    (w_a,) = set(ws_a)
    (w_b,) = set(ws_b)

    # w in [0, cutoff]
    assert 0 < w_a < potential.cutoff
    assert 0 < w_b < potential.cutoff

    if lamb < 0.5:
        assert w_a < w_b
    else:
        assert w_b < w_a


@pytest.mark.parametrize("host_system_fixture", ["solvent_host_system", "complex_host_system"])
def test_nonbonded_host_params_independent_of_lambda(
    host_system_fixture, hif2a_ligand_pair_single_topology: SingleTopology, request
):
    st = hif2a_ligand_pair_single_topology
    (host_sys, _), num_water_atoms, omm_topology = request.getfixturevalue(host_system_fixture)

    @jax.jit
    def get_nonbonded_host_params(lamb):
        return st.combine_with_host(host_sys, lamb, num_water_atoms, st.ff, omm_topology).nonbonded_host.params

    params0 = get_nonbonded_host_params(0.0)
    for lamb in np.linspace(0.1, 1, 10):
        params = get_nonbonded_host_params(lamb)
        np.testing.assert_array_equal(params, params0)
