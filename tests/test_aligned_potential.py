from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from common import get_alchemical_guest_params
from numpy.typing import NDArray

from timemachine.constants import DEFAULT_ATOM_MAPPING_KWARGS, NBParamIdx
from timemachine.fe import atom_mapping
from timemachine.fe.single_topology import AtomMapMixin, SingleTopology
from timemachine.fe.topology import get_ligand_ixn_pots_params
from timemachine.fe.utils import get_romol_conf, read_sdf_mols_by_name
from timemachine.ff import Forcefield
from timemachine.md.builders import build_protein_system
from timemachine.potentials import BoundPotential, Nonbonded, NonbondedInteractionGroup
from timemachine.utils import path_to_internal_file


def _parameterize_host_nonbonded_reference(
    num_guest_atoms, host_nonbonded: BoundPotential[Nonbonded]
) -> BoundPotential[Nonbonded]:
    """Parameterize host-host nonbonded interactions"""
    num_host_atoms = host_nonbonded.params.shape[0]
    host_params = host_nonbonded.params
    cutoff = host_nonbonded.potential.cutoff
    beta = host_nonbonded.potential.beta

    exclusion_idxs = host_nonbonded.potential.exclusion_idxs
    scale_factors = host_nonbonded.potential.scale_factors

    # Note: The choice of zeros here is arbitrary. It doesn't affect the
    # potentials or grads, but any function like the seed could depend on these values.
    hg_nb_params = jnp.concatenate([host_params, np.zeros((num_guest_atoms, host_params.shape[1]))])

    combined_nonbonded = Nonbonded(
        num_host_atoms + num_guest_atoms,
        exclusion_idxs,
        scale_factors,
        beta,
        cutoff,
        atom_idxs=np.arange(num_host_atoms, dtype=np.int32),
    )

    return combined_nonbonded.bind(hg_nb_params)


def _parameterize_host_guest_nonbonded_ixn_reference(
    mol_a,
    mol_b,
    atom_map_mixin,
    lamb: float,
    host_nonbonded: BoundPotential[Nonbonded],
    num_water_atoms: int,
    ff: Forcefield,
    omm_topology,
) -> BoundPotential[NonbondedInteractionGroup]:
    """Parameterize nonbonded interactions between the host and guest"""

    num_host_atoms = host_nonbonded.potential.num_atoms
    cutoff = host_nonbonded.potential.cutoff

    guest_ixn_env_params = get_alchemical_guest_params(
        mol_a, mol_b, atom_map_mixin, ff.q_handle, ff.lj_handle, lamb, cutoff
    )

    # L-W terms
    num_other_atoms = num_host_atoms - num_water_atoms

    num_guest_atoms = atom_map_mixin.get_num_atoms()

    def get_lig_idxs() -> NDArray[np.int32]:
        return np.arange(num_guest_atoms, dtype=np.int32) + num_host_atoms

    def get_water_idxs() -> NDArray[np.int32]:
        return np.arange(num_water_atoms, dtype=np.int32) + num_other_atoms

    def get_other_idxs() -> NDArray[np.int32]:
        return np.arange(num_other_atoms, dtype=np.int32)

    def get_env_idxs() -> NDArray[np.int32]:
        return np.concatenate([get_other_idxs(), get_water_idxs()])

    hg_nb_ixn_params = host_nonbonded.params.copy()
    if ff.env_bcc_handle is not None:
        env_bcc_h = ff.env_bcc_handle.get_env_handle(omm_topology, ff)
        hg_nb_ixn_params[:, NBParamIdx.Q_IDX] = env_bcc_h.parameterize(ff.env_bcc_handle.params)

    ixn_pot, ixn_params = get_ligand_ixn_pots_params(
        get_lig_idxs(),
        get_env_idxs(),
        hg_nb_ixn_params,
        guest_ixn_env_params,
        beta=host_nonbonded.potential.beta,
        cutoff=cutoff,
    )

    bound_ixn_pot = ixn_pot.bind(ixn_params)
    return bound_ixn_pot


def _get_core_by_mcs(mol_a, mol_b):
    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        **DEFAULT_ATOM_MAPPING_KWARGS,
    )

    core = all_cores[0]
    return core


def get_test_system():
    with path_to_internal_file("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols = read_sdf_mols_by_name(path_to_ligand)

    mol_a = mols["338"]
    mol_b = mols["43"]
    # Test that the nonbonded interaction group returns identical results as the current SingleTopology implementation.

    ff = Forcefield.load_default()

    with path_to_internal_file("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        host_config = build_protein_system(str(path_to_pdb), ff.protein_ff, ff.water_ff)
        host_config.box += np.diag([0.1, 0.1, 0.1])

    core = _get_core_by_mcs(mol_a, mol_b)

    st = SingleTopology.from_mols_with_host(mol_a, mol_b, core, host_config, ff)

    return mol_a, mol_b, core, st, host_config, ff


def test_nonbonded_interaction_group():
    mol_a, mol_b, core, st, host_config, ff = get_test_system()
    guest_atom_map_mixin = AtomMapMixin(mol_a.GetNumAtoms(), mol_b.GetNumAtoms(), core)
    box0 = host_config.box

    for lamb in np.linspace(0, 1, 12):
        test_ixn_group = st.setup_intermediate_state(lamb).nonbonded_ixn_group
        ref_ixn_group = _parameterize_host_guest_nonbonded_ixn_reference(
            mol_a,
            mol_b,
            guest_atom_map_mixin,
            lamb,
            host_config.host_system.nonbonded_all_pairs,
            host_config.num_water_atoms,
            ff,
            host_config.omm_topology,
        )

        lhs_conf = np.vstack([host_config.conf, get_romol_conf(mol_a)])
        rhs_conf = np.vstack([host_config.conf, get_romol_conf(mol_b)])
        coords = st.combine_confs(lhs_conf, rhs_conf, lamb)
        np.testing.assert_array_equal(ref_ixn_group.params, test_ixn_group.params)
        np.testing.assert_array_equal(ref_ixn_group.potential.row_atom_idxs, test_ixn_group.potential.row_atom_idxs)
        np.testing.assert_array_equal(ref_ixn_group.potential.col_atom_idxs, test_ixn_group.potential.col_atom_idxs)
        ref_u = ref_ixn_group(coords, box0)
        test_u = test_ixn_group(coords, box0)
        np.testing.assert_equal(ref_u, test_u)
        # gpu version
        ref_du_dx, ref_u = ref_ixn_group.to_gpu(np.float32).bound_impl.execute(coords, box0)
        test_du_dx, test_u = test_ixn_group.to_gpu(np.float32).bound_impl.execute(coords, box0)

        np.testing.assert_equal(ref_u, test_u)
        np.testing.assert_array_equal(ref_du_dx, test_du_dx)


def test_nonbonded_all_pairs():
    mol_a, mol_b, core, st, host_config, ff = get_test_system()
    guest_atom_map_mixin = AtomMapMixin(mol_a.GetNumAtoms(), mol_b.GetNumAtoms(), core)
    box0 = host_config.box

    for lamb in np.linspace(0, 1, 12):
        test_nb_ap = st.setup_intermediate_state(lamb).nonbonded_all_pairs
        ref_nb_ap = _parameterize_host_nonbonded_reference(
            guest_atom_map_mixin.get_num_atoms(), host_config.host_system.nonbonded_all_pairs
        )

        lhs_conf = np.vstack([host_config.conf, get_romol_conf(mol_a)])
        rhs_conf = np.vstack([host_config.conf, get_romol_conf(mol_b)])
        coords = st.combine_confs(lhs_conf, rhs_conf, lamb)

        np.testing.assert_array_equal(ref_nb_ap.potential.num_atoms, test_nb_ap.potential.num_atoms)
        np.testing.assert_array_equal(ref_nb_ap.params, test_nb_ap.params)
        np.testing.assert_array_equal(ref_nb_ap.potential.atom_idxs, test_nb_ap.potential.atom_idxs)
        np.testing.assert_array_equal(ref_nb_ap.potential.exclusion_idxs, test_nb_ap.potential.exclusion_idxs)
        np.testing.assert_array_equal(ref_nb_ap.potential.scale_factors, test_nb_ap.potential.scale_factors)

        # a bit redundant, since if all the parameters are identical between the two BPs, all downstream calculations
        # should also be identical.
        # reference version
        ref_u = ref_nb_ap(coords, box0)
        test_u = test_nb_ap(coords, box0)
        np.testing.assert_equal(ref_u, test_u)

        # gpu version
        ref_du_dx, ref_u = ref_nb_ap.to_gpu(np.float32).bound_impl.execute(coords, box0)
        test_du_dx, test_u = test_nb_ap.to_gpu(np.float32).bound_impl.execute(coords, box0)

        np.testing.assert_equal(ref_u, test_u)
        np.testing.assert_array_equal(ref_du_dx, test_du_dx)
