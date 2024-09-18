import numpy as np
import pytest

from timemachine.fe.free_energy import HostConfig
from timemachine.fe.rbfe import setup_optimized_host
from timemachine.fe.single_topology import SingleTopology
from timemachine.fe.single_topology_rest import SingleTopologyREST
from timemachine.fe.system import VacuumSystem
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import Forcefield
from timemachine.md import builders
from timemachine.potentials import PeriodicTorsion
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


@pytest.fixture(scope="module")
def hif2a_example_single_topology():
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_default()
    return SingleTopology(mol_a, mol_b, core, ff)


@pytest.fixture(scope="module")
def hif2a_example_vacuum(hif2a_example_single_topology):
    st = hif2a_example_single_topology
    st_rest = SingleTopologyREST(st.mol_a, st.mol_b, st.core, st.ff)

    mol_a_conf = get_romol_conf(st.mol_a)
    mol_b_conf = get_romol_conf(st.mol_b)

    ligand_conf = st.combine_confs(mol_a_conf, mol_b_conf)
    ligand_conf_ref = st.combine_confs(mol_a_conf, mol_b_conf)
    np.testing.assert_array_equal(ligand_conf, ligand_conf_ref)

    return st_rest, st, ligand_conf


@pytest.mark.parametrize("lamb", [0.0, 0.4, 0.5, 1.0])
def test_single_topology_rest_vacuum(hif2a_example_vacuum, lamb):
    st_rest, st, ligand_conf = hif2a_example_vacuum

    state = st_rest.setup_intermediate_state(lamb)
    state_ref = st.setup_intermediate_state(lamb)

    U = state.get_U_fn()(ligand_conf)
    U_ref = state_ref.get_U_fn()(ligand_conf)

    assert U == U_ref if lamb == 0.0 or lamb == 1.0 else U < U_ref

    scale = st_rest.get_rest_energy_scale_factor(lamb)

    np.testing.assert_allclose(
        state.nonbonded(ligand_conf, None),
        scale * state_ref.nonbonded(ligand_conf, None),
        rtol=1e-5,
    )

    assert state.torsion
    assert state_ref.torsion
    U_torsion = state.torsion(ligand_conf, None)
    U_torsion_ref = state_ref.torsion(ligand_conf, None)

    assert U_torsion == U_torsion_ref if lamb == 0.0 or lamb == 1.0 else U_torsion < U_torsion_ref

    def get_rest_torsion_potential(state: VacuumSystem):
        assert state.torsion
        mask = (state.torsion.potential.idxs[:, None] == st_rest._rest_torsions[None, :]).all(-1).any(-1)
        idxs = state.torsion.potential.idxs[mask]
        params = state.torsion.params[mask]
        return PeriodicTorsion(idxs).bind(params)

    U_rest_torsion = get_rest_torsion_potential(state)
    U_rest_torsion_ref = get_rest_torsion_potential(state_ref)

    np.testing.assert_allclose(U_rest_torsion(ligand_conf, None), scale * U_rest_torsion_ref(ligand_conf, None))


@pytest.fixture(scope="module")
def hif2a_example_solvent(hif2a_example_vacuum):
    st_rest, st, ligand_conf = hif2a_example_vacuum

    def get_solvent_host_config(box_width=4.0):
        solvent_sys, solvent_conf, solvent_box, _ = builders.build_water_system(
            box_width, st.ff.water_ff, mols=[st.mol_a, st.mol_b]
        )
        solvent_box += np.diag([0.1, 0.1, 0.1])
        return HostConfig(solvent_sys, solvent_conf, solvent_box, solvent_conf.shape[0])

    host_config = get_solvent_host_config()

    host = setup_optimized_host(st, host_config)
    host_ref = setup_optimized_host(st, host_config)
    np.testing.assert_array_equal(host.conf, host_ref.conf)
    np.testing.assert_array_equal(host.box, host_ref.box)

    return st_rest, st, ligand_conf, host, host_config


@pytest.mark.parametrize("lamb", [0.0, 0.4, 0.5, 1.0])
def test_single_topology_rest_solvent(hif2a_example_solvent, lamb):
    st_rest, st, ligand_conf, host, host_config = hif2a_example_solvent
    conf = np.concatenate([host.conf, ligand_conf])
    box = host_config.box

    def get_nonbonded_host_guest_ixn_potential(st: SingleTopology, lamb: float):
        hgs = st.combine_with_host(host.system, lamb, host_config.num_water_atoms)
        return hgs.nonbonded_host_guest_ixn

    U_fn = get_nonbonded_host_guest_ixn_potential(st_rest, lamb)
    U = U_fn(conf, box)

    U_fn_ref = get_nonbonded_host_guest_ixn_potential(st, lamb)
    U_ref = U_fn_ref(conf, box)

    scale = st_rest.get_rest_energy_scale_factor(lamb)
    np.testing.assert_allclose(U, scale * U_ref, rtol=1e-5)
