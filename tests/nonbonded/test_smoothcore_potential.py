import numpy as np
from common import GradientTest

from timemachine.fe.free_energy import HostConfig
from timemachine.fe.single_topology import AtomMapFlags, AtomMapMixin, SingleTopology
from timemachine.fe.system import convert_bps_into_system
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.md.builders import build_protein_system
from timemachine.md.minimizer import pre_equilibrate_host
from timemachine.potentials import SmoothcoreNonbondedInteractionGroup
from timemachine.potentials.nonbonded import (
    MIN_SMOOTHCORE_EPS_SQRT,
    MIN_SMOOTHCORE_SIG_HALF,
    reference_nonbonded_interaction_group,
    smoothcore_charge_interpolation,
    smoothcore_lj_interpolation,
)
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


def get_hif2a_protein_coords_box_params(mol_a, mol_b, ff, minimize=False):
    solvated_host_system, solvated_host_coords, box, topology, num_water_atoms = build_protein_system(
        "timemachine/testsystems/data/hif2a_nowater_min.pdb", ff.protein_ff, ff.water_ff, [mol_a, mol_b]
    )

    host_bps, _ = openmm_deserializer.deserialize_system(solvated_host_system, cutoff=1.2)

    host_system = convert_bps_into_system(host_bps)

    if minimize:
        hc = HostConfig(solvated_host_system, solvated_host_coords, box, num_water_atoms)
        equil_coords, equil_box = pre_equilibrate_host([mol_a, mol_b], hc, ff)
        assert equil_coords.shape == solvated_host_coords.shape
        return equil_coords, equil_box, host_bps[-1].params, host_system, num_water_atoms
    else:
        return solvated_host_coords, box, host_bps[-1].params, host_system, num_water_atoms


def test_smoothcore_charge_interpolation():
    """
    Test that:
    1) charge interpolation along the schedule maintains net-neutral charges
    2) end-states are consistent with the interpolate lambda values of 0 and 1
    """
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()

    ff = Forcefield.load_default()

    mol_a_charges = ff.q_handle.parameterize(mol_a)
    mol_b_charges = ff.q_handle.parameterize(mol_b)
    amm = AtomMapMixin(mol_a, mol_b, core)

    charges_src = np.zeros(amm.get_num_atoms())
    charges_dst = np.zeros(amm.get_num_atoms())

    mol_a_idxs = amm.mol_a_idxs()
    mol_b_idxs = amm.mol_b_idxs()

    charges_src[mol_a_idxs] = mol_a_charges[np.array([amm.c_to_a[x] for x in mol_a_idxs])]
    charges_dst[mol_b_idxs] = mol_b_charges[np.array([amm.c_to_b[x] for x in mol_b_idxs])]

    # test that net charges are zero through-out the schedule.
    for lamb in np.linspace(0, 1.0, 20):
        charges = smoothcore_charge_interpolation(lamb, charges_src, charges_dst, mol_a_idxs, mol_b_idxs)
        np.testing.assert_almost_equal(np.sum(charges), 0, decimal=5)

    charges_lhs = smoothcore_charge_interpolation(0.0, charges_src, charges_dst, mol_a_idxs, mol_b_idxs)
    charges_rhs = smoothcore_charge_interpolation(1.0, charges_src, charges_dst, mol_a_idxs, mol_b_idxs)

    np.testing.assert_array_equal(charges_lhs, charges_src)
    np.testing.assert_array_equal(charges_rhs, charges_dst)


def test_smoothcore_lj_interpolation():
    """
    Test that the lj interpolation has the correct truncation treatment (upon absorption into
    the anchoring atoms).
    """
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()

    ff = Forcefield.load_default()

    mol_a_lj = ff.lj_handle.parameterize(mol_a)
    mol_b_lj = ff.lj_handle.parameterize(mol_b)
    amm = AtomMapMixin(mol_a, mol_b, core)

    lj_src = np.zeros((amm.get_num_atoms(), 2))
    lj_dst = np.zeros((amm.get_num_atoms(), 2))

    mol_a_idxs = amm.mol_a_idxs()
    mol_b_idxs = amm.mol_b_idxs()

    lj_src[mol_a_idxs] = mol_a_lj[np.array([amm.c_to_a[x] for x in mol_a_idxs])]
    lj_dst[mol_b_idxs] = mol_b_lj[np.array([amm.c_to_b[x] for x in mol_b_idxs])]

    lj_lhs = smoothcore_lj_interpolation(0.0, lj_src, lj_dst, mol_a_idxs, mol_b_idxs)
    lj_rhs = smoothcore_lj_interpolation(1.0, lj_src, lj_dst, mol_a_idxs, mol_b_idxs)

    np.testing.assert_array_equal(lj_lhs, lj_src)
    np.testing.assert_array_equal(lj_rhs, lj_dst)
    assert np.all(lj_lhs[amm.dummy_b_idxs(), 0] == 0)
    assert np.all(lj_lhs[amm.dummy_b_idxs(), 1] == 0)
    assert np.all(lj_rhs[amm.dummy_a_idxs(), 0] == 0)
    assert np.all(lj_rhs[amm.dummy_a_idxs(), 1] == 0)

    lj_almost_lhs = smoothcore_lj_interpolation(0.0001, lj_src, lj_dst, mol_a_idxs, mol_b_idxs)
    assert np.all(lj_almost_lhs[amm.dummy_b_idxs(), 0] >= MIN_SMOOTHCORE_SIG_HALF)
    assert np.all(lj_almost_lhs[amm.dummy_b_idxs(), 1] >= MIN_SMOOTHCORE_EPS_SQRT)

    lj_almost_rhs = smoothcore_lj_interpolation(0.9999, lj_src, lj_dst, mol_a_idxs, mol_b_idxs)
    assert np.all(lj_almost_rhs[amm.dummy_a_idxs(), 0] >= MIN_SMOOTHCORE_SIG_HALF)
    assert np.all(lj_almost_rhs[amm.dummy_a_idxs(), 1] >= MIN_SMOOTHCORE_EPS_SQRT)


def test_smoothcore_ligand_anchor_idxs():
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, ff)

    ligand_anchor_idxs = np.zeros(st.get_num_atoms(), dtype=np.int32)
    for c_idx, c_flag in enumerate(st.c_flags):
        if c_flag == AtomMapFlags.CORE:
            ligand_anchor_idxs[c_idx] = c_idx

    st = SingleTopology(mol_a, mol_b, core, ff)

    # anchors implied by the atom-mapping, manually generated by means of eye-balling
    ligand_anchor_idxs[st.a_to_c[22]] = st.a_to_c[0]
    ligand_anchor_idxs[st.a_to_c[24]] = st.a_to_c[0]
    ligand_anchor_idxs[st.a_to_c[25]] = st.a_to_c[0]
    ligand_anchor_idxs[st.a_to_c[33]] = st.a_to_c[0]
    ligand_anchor_idxs[st.b_to_c[23]] = st.b_to_c[22]

    test_ligand_anchor_idxs = st.get_smoothcore_ligand_anchor_idxs()

    np.testing.assert_array_equal(test_ligand_anchor_idxs, ligand_anchor_idxs)


def test_smoothcore_potential():
    """
    Test that the smooth core potential on the minimized hif2a system.

    Verifies:

    1) Agreement with reference jax implementation
    2) Agreement with existing NonbondedInteractionGroup at the end-states

    """

    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    x_a = get_romol_conf(mol_a)
    x_b = get_romol_conf(mol_b)

    ff = Forcefield.load_default()
    host_conf, box, host_params, host_system, num_water_atoms = get_hif2a_protein_coords_box_params(
        mol_a, mol_b, ff, minimize=True
    )

    ref_beta = host_system.nonbonded.potential.beta
    ref_cutoff = host_system.nonbonded.potential.cutoff

    num_host_atoms = len(host_conf)

    st = SingleTopology(mol_a, mol_b, core, ff, use_smoothcore=True)
    row_atom_idxs = np.arange(st.get_num_atoms()) + num_host_atoms
    col_atom_idxs = np.arange(num_host_atoms)

    ligand_anchor_idxs = st.get_smoothcore_ligand_anchor_idxs()
    ligand_anchor_idxs += num_host_atoms
    anchor_idxs = np.concatenate([np.arange(num_host_atoms), ligand_anchor_idxs])

    xs = []
    test_nrgs = []
    test_params = []

    total_num_atoms = num_host_atoms + st.get_num_atoms()
    ref_pot = SmoothcoreNonbondedInteractionGroup(
        total_num_atoms,
        row_atom_idxs.astype(np.int32),
        anchor_idxs.astype(np.int32),
        ref_beta,
        ref_cutoff,
        col_atom_idxs.astype(np.int32),
    )

    for lamb in np.linspace(0, 1, 11):
        ligand_params = st._get_smoothcore_guest_params(ff.q_handle, ff.lj_handle, lamb)
        combined_params = np.concatenate([host_params, ligand_params])
        ligand_conf = st.combine_confs(x_a, x_b, lamb)
        combined_conf = np.concatenate([host_conf, ligand_conf])
        test_params.append(combined_params)
        ref_nrg = ref_pot(combined_conf, combined_params, box)
        xs.append(combined_conf)
        test_nrgs.append(ref_nrg)

        for precision, rtol, atol in [(np.float64, 1e-8, 1e-8), (np.float32, 5e-4, 5e-4)]:
            test_potential, combined_params = st._parameterize_unbound_host_guest_nonbonded_ixn(
                lamb,
                host_system.nonbonded,
                num_water_atoms,
            )
            test_impl = test_potential.to_gpu(precision)
            GradientTest().compare_forces(combined_conf, combined_params, box, ref_pot, test_impl, rtol=rtol, atol=atol)

    # test that reference gpu, reference cpu, and the smoothcore are consistent at the end-states
    st = SingleTopology(mol_a, mol_b, core, ff, use_smoothcore=False)
    hg_system_lhs = st.combine_with_host(host_system, 0.0, num_water_atoms)
    ref_u_lhs_gpu = hg_system_lhs.nonbonded_host_guest_ixn(xs[0], box)
    ref_params_lhs = hg_system_lhs.nonbonded_host_guest_ixn.params.reshape(-1, 4)
    ref_u_lhs_cpu = reference_nonbonded_interaction_group(
        xs[0], ref_params_lhs, box, row_atom_idxs, col_atom_idxs, beta=ref_beta, cutoff=ref_cutoff
    )
    np.testing.assert_allclose(test_nrgs[0], ref_u_lhs_cpu)
    np.testing.assert_allclose(ref_u_lhs_gpu, ref_u_lhs_cpu)

    hg_system_rhs = st.combine_with_host(host_system, 1.0, num_water_atoms)
    ref_u_rhs_gpu = hg_system_rhs.nonbonded_host_guest_ixn(xs[-1], box)
    ref_params_rhs = hg_system_rhs.nonbonded_host_guest_ixn.params.reshape(-1, 4)
    ref_u_rhs_cpu = reference_nonbonded_interaction_group(
        xs[-1], ref_params_rhs, box, row_atom_idxs, col_atom_idxs, beta=ref_beta, cutoff=ref_cutoff
    )
    np.testing.assert_allclose(test_nrgs[-1], ref_u_rhs_cpu)
    np.testing.assert_allclose(ref_u_rhs_gpu, ref_u_rhs_cpu)
