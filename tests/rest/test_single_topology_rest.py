from functools import cache

import jax
import matplotlib.pyplot as plt
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from timemachine.constants import DEFAULT_ATOM_MAPPING_KWARGS
from timemachine.fe import atom_mapping
from timemachine.fe.free_energy import HostConfig
from timemachine.fe.rbfe import Host, setup_optimized_host
from timemachine.fe.rest.interpolation import (
    Exponential,
    InterpolationFxnName,
    Linear,
    Quadratic,
    Symmetric,
    plot_interpolation_fxn,
)
from timemachine.fe.rest.single_topology import SingleTopologyREST
from timemachine.fe.single_topology import SingleTopology
from timemachine.fe.system import GuestSystem
from timemachine.fe.utils import get_romol_conf, read_sdf_mols_by_name
from timemachine.ff import Forcefield
from timemachine.md import builders
from timemachine.potentials import PeriodicTorsion
from timemachine.utils import path_to_internal_file

with path_to_internal_file("timemachine.testsystems.fep_benchmark.hif2a", "ligands.sdf") as ligands_path:
    hif2a_ligands = read_sdf_mols_by_name(ligands_path)

hif2a_ligand_pairs = [
    (mol_a, mol_b)
    for mol_a_name, mol_a in hif2a_ligands.items()
    for mol_b_name, mol_b in hif2a_ligands.items()
    if mol_a_name < mol_b_name
]

forcefield = Forcefield.load_default()


@cache
def get_core(mol_a, mol_b) -> tuple[tuple[int, int], ...]:
    core_array = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)[0]
    return tuple((a, b) for a, b in core_array)


@cache
def get_single_topology(mol_a, mol_b, core) -> SingleTopology:
    return SingleTopology(mol_a, mol_b, np.asarray(core), forcefield)


@cache
def get_single_topology_rest(
    mol_a, mol_b, core, max_temperature_scale: float, temperature_scale_interpolation_fxn: InterpolationFxnName
) -> SingleTopologyREST:
    return SingleTopologyREST(
        mol_a, mol_b, np.asarray(core), forcefield, max_temperature_scale, temperature_scale_interpolation_fxn
    )


@pytest.mark.parametrize("lamb", [0.0, 0.4, 0.5, 1.0])
@pytest.mark.parametrize("temperature_scale_interpolation_fxn", ["linear", "quadratic", "exponential"])
@pytest.mark.parametrize("mol_pair", np.random.default_rng(2024).choice(hif2a_ligand_pairs, size=3))
def test_single_topology_rest_vacuum(mol_pair, temperature_scale_interpolation_fxn, lamb):
    mol_a, mol_b = mol_pair

    has_aliphatic_rings = (
        rdMolDescriptors.CalcNumAliphaticRings(mol_a) > 0 or rdMolDescriptors.CalcNumAliphaticRings(mol_b) > 0
    )
    has_rotatable_bonds = (
        rdMolDescriptors.CalcNumRotatableBonds(mol_a) > 0 or rdMolDescriptors.CalcNumRotatableBonds(mol_b) > 0
    )

    core = get_core(mol_a, mol_b)
    st = get_single_topology(mol_a, mol_b, core)
    st_rest = get_single_topology_rest(mol_a, mol_b, core, 2.0, temperature_scale_interpolation_fxn)

    state = st_rest.setup_intermediate_state(lamb)
    state_ref = st.setup_intermediate_state(lamb)
    assert len(st_rest.candidate_propers) < len(state_ref.proper.potential.idxs)

    ligand_conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))

    U_proper = state.proper(ligand_conf, None)
    U_proper_ref = state_ref.proper(ligand_conf, None)

    U_nonbonded = state.nonbonded_pair_list(ligand_conf, None)
    U_nonbonded_ref = state_ref.nonbonded_pair_list(ligand_conf, None)

    U = state.get_U_fn()(ligand_conf)
    U_ref = state_ref.get_U_fn()(ligand_conf)

    energy_scale = st_rest.get_energy_scale_factor(lamb)

    if lamb == 0.0 or lamb == 1.0:
        assert energy_scale == 1.0

        assert U_proper == U_proper_ref
        assert U_nonbonded == U_nonbonded_ref
        assert U == U_ref

    else:
        assert energy_scale < 1.0

        if has_rotatable_bonds or has_aliphatic_rings:
            assert 0 < len(st_rest.candidate_propers)

            if energy_scale < 1.0:
                assert U_proper < U_proper_ref

        def get_proper_subset_energy(state: GuestSystem, ixn_idxs):
            assert state.proper
            idxs = state.proper.potential.idxs[ixn_idxs, :]
            params = state.proper.params[ixn_idxs, :]
            potential = PeriodicTorsion(idxs).bind(params)
            return potential(ligand_conf, None)

        U_proper_subset = get_proper_subset_energy(state, st_rest.target_proper_idxs)
        U_proper_subset_ref = get_proper_subset_energy(state_ref, st_rest.target_proper_idxs)
        np.testing.assert_allclose(U_proper_subset, energy_scale * U_proper_subset_ref)

        np.testing.assert_allclose(U_nonbonded, energy_scale * U_nonbonded_ref)


@cache
def get_solvent_host(st: SingleTopology) -> tuple[Host, HostConfig]:
    box_width = 4.0
    host_config = builders.build_water_system(box_width, forcefield.water_ff, mols=[st.mol_a, st.mol_b])
    host_config.box += np.diag([0.1, 0.1, 0.1])
    host = setup_optimized_host(st, host_config)

    return host, host_config


@pytest.mark.parametrize("lamb", [0.0, 0.4, 0.5, 1.0])
@pytest.mark.parametrize("temperature_scale_interpolation_fxn", ["linear", "quadratic", "exponential"])
@pytest.mark.parametrize("mol_pair", np.random.default_rng(2024).choice(hif2a_ligand_pairs, size=3))
def test_single_topology_rest_solvent(mol_pair, temperature_scale_interpolation_fxn, lamb):
    mol_a, mol_b = mol_pair

    core = get_core(mol_a, mol_b)
    st = get_single_topology(mol_a, mol_b, core)
    st_rest = get_single_topology_rest(mol_a, mol_b, core, 2.0, temperature_scale_interpolation_fxn)

    host, host_config = get_solvent_host(st)

    ligand_conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))
    conf = np.concatenate([host.conf, ligand_conf])

    def get_nonbonded_host_guest_ixn_energy(st: SingleTopology):
        hgs = st.combine_with_host(host.system, lamb, host_config.num_water_atoms, st.ff, host_config.omm_topology)
        return hgs.nonbonded_ixn_group(conf, host_config.box)

    U = get_nonbonded_host_guest_ixn_energy(st_rest)
    U_ref = get_nonbonded_host_guest_ixn_energy(st)

    energy_scale = st_rest.get_energy_scale_factor(lamb)
    np.testing.assert_allclose(U, energy_scale * U_ref, rtol=1e-5)


def get_mol(smiles: str):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=2024)
    return mol


def get_identity_transformation(mol):
    n_atoms = mol.GetNumAtoms()
    core = np.tile(np.arange(n_atoms)[:, None], (1, 2))  # identity
    return SingleTopologyREST(mol, mol, core, forcefield, 2.0, "linear")


def test_single_topology_rest_propers():
    # benzene: no propers are scaled
    benzene = get_mol("c1ccccc1")
    st = get_identity_transformation(benzene)
    assert st.candidate_propers == set()

    # cyclohexane: all 9 * 6 ring propers are scaled (|{H1, H2, C1}-C2-C3-{C4, H3, H4}| = 9 propers per C-C bond)
    cyclohexane = get_mol("C1CCCCC1")
    st = get_identity_transformation(cyclohexane)
    assert len({t for _, t in st.candidate_propers}) == 9 * 6

    # phenylcyclohexane: all 9 * 6 cyclohexane ring propers and 6 rotatable bond propers are scaled
    phenylcyclohexane = get_mol("c1ccc(C2CCCCC2)cc1")
    st = get_identity_transformation(phenylcyclohexane)
    assert len({t for _, t in st.candidate_propers}) == 9 * 6 + 6


@pytest.mark.parametrize(
    "lamb", [0.0, 0.4, 0.51, 1.0]
)  # NOTE: asymmetry at lambda = 0.5 due to discontinuity in combine_confs
@pytest.mark.parametrize("temperature_scale_interpolation_fxn", ["linear", "quadratic", "exponential"])
@pytest.mark.parametrize("mol_pair", np.random.default_rng(2024).choice(hif2a_ligand_pairs, size=3))
def test_single_topology_rest_symmetric(mol_pair, temperature_scale_interpolation_fxn, lamb):
    mol_a, mol_b = mol_pair
    core_fwd = get_core(mol_a, mol_b)
    core_rev = tuple((b, a) for a, b in core_fwd)

    def get_transformation(mol_a, mol_b, core, lamb):
        st = get_single_topology_rest(mol_a, mol_b, core, 2.0, temperature_scale_interpolation_fxn)
        potential = st.setup_intermediate_state(lamb).get_U_fn()
        conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b), lamb)
        return potential, conf, st

    u_fwd, conf_fwd, st_fwd = get_transformation(mol_a, mol_b, core_fwd, lamb)
    u_rev, conf_rev, st_rev = get_transformation(mol_b, mol_a, core_rev, 1.0 - lamb)

    assert len(st_fwd.rotatable_bonds) == len(st_rev.rotatable_bonds)
    assert len(st_fwd.aliphatic_ring_bonds) == len(st_rev.aliphatic_ring_bonds)
    assert len(st_fwd.target_proper_idxs) == len(st_rev.target_proper_idxs)

    np.testing.assert_allclose(u_fwd(conf_fwd), u_rev(conf_rev))

    core_to_a, core_to_b = np.asarray(core_fwd).T
    core_map = [(x, y) for x, y in zip(st_fwd.a_to_c, st_rev.b_to_c)]
    dummy_map = [(x, y) for x, y in zip(st_fwd.b_to_c, st_rev.a_to_c) if x not in core_to_a and y not in core_to_b]
    fused_map = core_map + dummy_map
    p_fwd, p_rev = np.array(fused_map).T

    np.testing.assert_allclose(
        jax.grad(u_fwd)(conf_fwd)[p_fwd],
        jax.grad(u_rev)(conf_rev)[p_rev],
    )


def plot_interpolation_fxns():
    src, dst = 1.0, 3.0
    _ = plot_interpolation_fxn(Symmetric(Linear(src, dst)))
    _ = plot_interpolation_fxn(Symmetric(Quadratic(src, dst)))
    _ = plot_interpolation_fxn(Symmetric(Exponential(src, dst)))
    _ = plt.legend()
    plt.show()
