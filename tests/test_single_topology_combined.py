# test code for combining host guest systems and ensuring that parameters
# and lambda configurations are correct
import jax
import numpy as np
import pytest

from timemachine import potentials
from timemachine.fe.free_energy import HostConfig
from timemachine.fe.single_topology import AtomMapMixin, SingleTopology
from timemachine.fe.system import HostSystem
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.md import builders
from timemachine.testsystems.dhfr import get_dhfr_system
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology

pytestmark = [pytest.mark.nocuda]


@pytest.fixture(scope="module")
def hif2a_ligand_pair_single_topology():
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_default()
    return mol_a, mol_b, core, forcefield
    # return SingleTopology(mol_a, mol_b, core, forcefield)


@pytest.fixture(scope="module")
def complex_host_config():
    # (YTZ): we need to clean this up later, since it uses a pre-solvated xml file.
    host_sys_omm, host_top = get_dhfr_system()
    # Hardcoded to match 5dfr_solv_equil.pdb file
    num_water_atoms = 21069
    (bond, angle, proper, improper, nonbonded), masses = openmm_deserializer.deserialize_system(host_sys_omm, 1.2)
    host_system = HostSystem(bond=bond, angle=angle, proper=proper, improper=improper, nonbonded_all_pairs=nonbonded)
    host_conf = np.zeros((len(masses), 3), dtype=np.float64)  # un-used
    host_box = np.eye(3, dtype=np.float64)  # un-used
    return HostConfig(host_system, host_conf, host_box, num_water_atoms, host_top, masses)


@pytest.fixture(scope="module")
def solvent_host_config():
    ff = Forcefield.load_default()
    return builders.build_water_system(3.0, ff.water_ff)


def to_bond_set(all_idxs):
    if all_idxs is None:
        return set()
    bonds = set()
    for idxs in all_idxs:
        idxs = tuple([int(x) for x in idxs])
        bonds.add(idxs)
    return bonds


# we no longer guarantee that bond_idxs are spliced in order since alignment dict can scramble things arbitrarily
@pytest.mark.parametrize("lamb", [0.0, 1.0])
@pytest.mark.parametrize("host_config_fixture", ["solvent_host_config", "complex_host_config"])
def test_combined_parameters_bonded(host_config_fixture, lamb, hif2a_ligand_pair_single_topology, request):
    # test bonded parameters are correct at the end-states.
    # 1) we expected bonded idxs in the ligand to be shifted by num_host_atoms

    mol_a, mol_b, core, ff = hif2a_ligand_pair_single_topology
    host_config: HostConfig = request.getfixturevalue(host_config_fixture)
    num_host_atoms = len(host_config.masses)

    host_guest_st = SingleTopology.from_mols_with_host(mol_a, mol_b, core, host_config, ff)
    guest_st = SingleTopology.from_mols(mol_a, mol_b, core, ff)

    guest_system = guest_st.setup_intermediate_state(lamb)
    host_guest_system = host_guest_st.setup_intermediate_state(lamb)
    host_system = host_config.host_system

    def assert_idxs_consistency(guest_idxs, host_idxs, host_guest_idxs):
        guest_idxs = to_bond_set(guest_idxs)
        host_idxs = to_bond_set(host_idxs)
        host_guest_idxs = to_bond_set(host_guest_idxs)

        guest_idxs_incremented = set()
        for idxs in guest_idxs:
            guest_idxs_incremented.add(tuple([x + num_host_atoms for x in idxs]))

        np.testing.assert_array_equal(sorted(guest_idxs_incremented.union(host_idxs)), sorted(host_guest_idxs))

    assert_idxs_consistency(
        guest_system.bond.potential.idxs, host_system.bond.potential.idxs, host_guest_system.bond.potential.idxs
    )
    assert_idxs_consistency(
        guest_system.angle.potential.idxs, host_system.angle.potential.idxs, host_guest_system.angle.potential.idxs
    )
    assert_idxs_consistency(
        guest_system.proper.potential.idxs, host_system.proper.potential.idxs, host_guest_system.proper.potential.idxs
    )
    assert_idxs_consistency(
        guest_system.improper.potential.idxs,
        host_system.improper.potential.idxs,
        host_guest_system.improper.potential.idxs,
    )
    assert_idxs_consistency(guest_system.chiral_atom.potential.idxs, None, host_guest_system.chiral_atom.potential.idxs)
    assert_idxs_consistency(guest_system.chiral_bond.potential.idxs, None, host_guest_system.chiral_bond.potential.idxs)
    assert_idxs_consistency(
        guest_system.nonbonded_pair_list.potential.idxs, None, host_guest_system.nonbonded_pair_list.potential.idxs
    )


@pytest.mark.parametrize("lamb", [0.0, 1.0])
@pytest.mark.parametrize("host_config_fixture", ["solvent_host_config", "complex_host_config"])
def test_combined_parameters_nonbonded(host_config_fixture, lamb, hif2a_ligand_pair_single_topology, request):
    # test bonded and nonbonded parameters are correct at the end-states.
    # 1) we expected bonded idxs in the ligand to be shifted by num_host_atoms
    # 2) we expected nonbonded lambda_idxs to be shifted
    # 3) we expected nonbonded parameters on the core to be linearly interpolated

    mol_a, mol_b, ligand_core, ff = hif2a_ligand_pair_single_topology
    host_config = request.getfixturevalue(host_config_fixture)
    num_host_atoms = len(host_config.masses)
    st = SingleTopology.from_mols_with_host(mol_a, mol_b, ligand_core, host_config, ff)
    hgs = st.setup_intermediate_state(lamb)
    # check nonbonded terms
    # 1) ligand ixns should be omitted in hgs.nonbonded_host
    assert isinstance(hgs.nonbonded_all_pairs.potential, potentials.Nonbonded)
    assert hgs.nonbonded_all_pairs.potential.atom_idxs is not None
    assert set(hgs.nonbonded_all_pairs.potential.atom_idxs) == set(range(num_host_atoms))

    # 2) decoupling parameters for host-guest interactions
    # 2a) w offsets
    potential = hgs.nonbonded_ixn_group.potential
    params = hgs.nonbonded_ixn_group.params

    guest_atom_map_mixin = AtomMapMixin(mol_a.GetNumAtoms(), mol_b.GetNumAtoms(), ligand_core)

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
            indicator = guest_atom_map_mixin.c_flags[guest_atom_idx]
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
    mol_a_charges = ff.q_handle.parameterize(mol_a)
    mol_a_sig_eps = ff.lj_handle.parameterize(mol_a)

    mol_b_charges = ff.q_handle.parameterize(mol_b)
    mol_b_sig_eps = ff.lj_handle.parameterize(mol_b)

    for a_idx, (test_q, test_sig, test_eps, _) in enumerate(params):
        if a_idx < num_host_atoms:
            continue

        guest_atom_idx = a_idx - num_host_atoms
        indicator = guest_atom_map_mixin.c_flags[guest_atom_idx]

        # dummy atom qlj parameters are arbitrary (since they will be decoupled via lambda parameters)
        if indicator != 0:
            continue

        if lamb == 0.0:
            # should resemble mol_a at lambda=0
            ref_q = mol_a_charges[guest_atom_map_mixin.c_to_a[guest_atom_idx]]
            ref_sig, ref_eps = mol_a_sig_eps[guest_atom_map_mixin.c_to_a[guest_atom_idx]]

            assert ref_q == test_q
            assert test_sig == ref_sig
            assert test_eps == ref_eps

        elif lamb == 1.0:
            # should resemble mol_b at lambda=1
            ref_q = mol_b_charges[guest_atom_map_mixin.c_to_b[guest_atom_idx]]
            ref_sig, ref_eps = mol_b_sig_eps[guest_atom_map_mixin.c_to_b[guest_atom_idx]]

            assert ref_q == test_q
            assert test_sig == ref_sig
            assert test_eps == ref_eps


@pytest.mark.parametrize("lamb", np.random.default_rng(2022).uniform(0.01, 0.99, (10,)))
@pytest.mark.parametrize("host_config_fixture", ["solvent_host_config", "complex_host_config"])
def test_combined_parameters_nonbonded_intermediate(
    host_config_fixture, lamb, hif2a_ligand_pair_single_topology: SingleTopology, request
):
    mol_a, mol_b, core, ff = hif2a_ligand_pair_single_topology
    host_config = request.getfixturevalue(host_config_fixture)
    num_host_atoms = len(host_config.masses)

    st = SingleTopology.from_mols_with_host(mol_a, mol_b, core, host_config, ff)
    hgs = st.setup_intermediate_state(lamb)
    potential = hgs.nonbonded_ixn_group.potential
    params = hgs.nonbonded_ixn_group.params
    assert isinstance(potential, potentials.NonbondedInteractionGroup)

    guest_atom_map_mixin = AtomMapMixin(mol_a.GetNumAtoms(), mol_b.GetNumAtoms(), core)

    guest_params = np.array(params[num_host_atoms:])
    ws_core = [w for flag, (_, _, _, w) in zip(guest_atom_map_mixin.c_flags, guest_params) if flag == 0]
    ws_a = [w for flag, (_, _, _, w) in zip(guest_atom_map_mixin.c_flags, guest_params) if flag == 1]
    ws_b = [w for flag, (_, _, _, w) in zip(guest_atom_map_mixin.c_flags, guest_params) if flag == 2]

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


@pytest.mark.parametrize("host_config_fixture", ["solvent_host_config", "complex_host_config"])
def test_nonbonded_host_params_independent_of_lambda(
    host_config_fixture, hif2a_ligand_pair_single_topology: SingleTopology, request
):
    mol_a, mol_b, core, ff = hif2a_ligand_pair_single_topology
    host_config = request.getfixturevalue(host_config_fixture)
    st = SingleTopology.from_mols_with_host(mol_a, mol_b, core, host_config, ff)

    @jax.jit
    def get_nonbonded_host_params(lamb):
        return st.setup_intermediate_state(lamb).nonbonded_all_pairs.params

    params0 = get_nonbonded_host_params(0.0)
    for lamb in np.linspace(0.1, 1, 10):
        params = get_nonbonded_host_params(lamb)
        np.testing.assert_array_equal(
            params.astype(np.float32).astype(np.float64), params0.astype(np.float32).astype(np.float64)
        )
