from functools import partial
from typing import no_type_check

import jax.numpy as jnp
import numpy as np
import pytest
from common import check_split_ixns
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine import potentials
from timemachine.constants import NBParamIdx
from timemachine.fe import topology
from timemachine.fe.topology import _SCALE_14_LJ, _SCALE_14_Q, BaseTopology, DualTopology
from timemachine.fe.utils import get_romol_conf, read_sdf, read_sdf_mols_by_name, set_romol_conf
from timemachine.ff import Forcefield
from timemachine.potentials.nonbonded import combining_rule_epsilon, combining_rule_sigma
from timemachine.utils import path_to_internal_file


@pytest.mark.nocuda
def test_base_topology_14_exclusions():
    with path_to_internal_file("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        all_mols = read_sdf(path_to_ligand)

    mol = all_mols[0]

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    bt = topology.BaseTopology(mol, ff)
    nb_params, nb = bt.parameterize_nonbonded_pairlist(
        ff.q_handle.params, ff.q_handle_intra.params, ff.lj_handle.params, ff.lj_handle_intra.params, True
    )

    kvs = dict()
    for (i, j), qljw in zip(nb.idxs, nb_params):
        kvs[(i, j)] = qljw

    qs = ff.q_handle.parameterize(mol)
    ljs = ff.lj_handle.parameterize(mol)

    sigmas = ljs[:, 0]
    epsilons = ljs[:, 1]

    # atoms 27-0-1-28 correspond to a H-O-C-H torsion (respectively), we expect:
    # q_ij to be rescaled by (1-SCALE_14_Q)
    # sigma_ij to be unscaled
    # eps_ij to be rescaled by (1-SCALE_14_LJ)
    np.testing.assert_almost_equal(kvs[(27, 28)][0], qs[27] * qs[28] * (1 - _SCALE_14_Q))
    np.testing.assert_almost_equal(kvs[(27, 28)][1], combining_rule_sigma(sigmas[27], sigmas[28]))
    np.testing.assert_almost_equal(
        kvs[(27, 28)][2], combining_rule_epsilon(epsilons[27], epsilons[28]) * (1 - _SCALE_14_LJ)
    )

    # 0-1-28 correspond to an O-C-H angle, we expect it to be missing
    assert (0, 28) not in kvs
    assert (28, 0) not in kvs


def parameterize_nonbonded_full(
    hgt: topology.HostGuestTopology,
    ff_q_params,
    ff_q_params_intra,
    ff_lj_params,
    ff_lj_params_intra,
    lamb: float,
):
    # Implements the full NB potential for the host guest system
    num_guest_atoms = hgt.guest_topology.get_num_atoms()
    guest_params, guest_pot = hgt.guest_topology.parameterize_nonbonded(
        ff_q_params, ff_q_params_intra, ff_lj_params, ff_lj_params_intra, lamb
    )
    assert hgt.host_nonbonded is not None
    hg_exclusion_idxs = np.concatenate(
        [hgt.host_nonbonded.potential.exclusion_idxs, guest_pot.exclusion_idxs + hgt.num_host_atoms]
    )
    hg_scale_factors = np.concatenate([hgt.host_nonbonded.potential.scale_factors, guest_pot.scale_factors])
    hg_nb_params = jnp.concatenate([hgt.host_nonbonded.params, guest_params])
    return hg_nb_params, potentials.Nonbonded(
        hgt.num_host_atoms + num_guest_atoms, hg_exclusion_idxs, hg_scale_factors, guest_pot.beta, guest_pot.cutoff
    )


@no_type_check
def test_host_guest_nonbonded_tiny_mol():
    ctor = BaseTopology
    precision, rtol, atol = (np.float32, 1e-4, 5e-4)
    use_tiny_mol = True
    host_guest_nonbonded_impl(ctor, precision, rtol, atol, use_tiny_mol)


@no_type_check
@pytest.mark.parametrize("precision, rtol, atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
@pytest.mark.parametrize("ctor", [BaseTopology, DualTopology])
@pytest.mark.parametrize("use_tiny_mol", [True, False])
@pytest.mark.nightly(reason="slow")
def test_host_guest_nonbonded(ctor, precision, rtol, atol, use_tiny_mol):
    host_guest_nonbonded_impl(ctor, precision, rtol, atol, use_tiny_mol)


@no_type_check
def host_guest_nonbonded_impl(ctor, precision, rtol, atol, use_tiny_mol):
    def compute_ref_grad_u(ff: Forcefield, precision, x0, box, lamb, num_water_atoms, host_system, omm_topology):
        # Use the original code to compute the nb grads and potential
        bt = Topology(ff)
        hgt = topology.HostGuestTopology(host_system.get_U_fns(), bt, num_water_atoms, ff, omm_topology)
        params, us = parameterize_nonbonded_full(
            hgt,
            ff.q_handle.params,
            ff.q_handle_intra.params,
            ff.lj_handle.params,
            ff.lj_handle_intra.params,
            lamb=lamb,
        )
        u_impl = us.bind(params).to_gpu(precision=precision).bound_impl
        return u_impl.execute(x0, box)

    def compute_new_grad_u(ff: Forcefield, precision, x0, box, lamb, num_water_atoms, host_system, omm_topology):
        # Use the updated topology code to compute the nb grads and potential
        bt = Topology(ff)
        hgt = topology.HostGuestTopology(host_system.get_U_fns(), bt, num_water_atoms, ff, omm_topology)
        params, us = hgt.parameterize_nonbonded(
            ff.q_handle.params,
            ff.q_handle_intra.params,
            ff.lj_handle.params,
            ff.lj_handle_intra.params,
            lamb=lamb,
        )
        u_impl = us.bind(params).to_gpu(precision=precision).bound_impl
        return u_impl.execute(x0, box)

    def compute_intra_grad_u(ff: Forcefield, precision, x0, box, lamb, num_water_atoms, num_host_atoms):
        # Compute the vacuum nb grads and potential for the ligand intramolecular term
        bt = Topology(ff)
        params, us = bt.parameterize_nonbonded(
            ff.q_handle.params,
            ff.q_handle_intra.params,
            ff.lj_handle.params,
            ff.lj_handle_intra.params,
            lamb=lamb,
        )
        u_impl = us.bind(params).to_gpu(precision=precision).bound_impl
        g, u = u_impl.execute(x0, box)

        # Pad g so it's the same shape as the others
        g_padded = np.concatenate([np.zeros((num_host_atoms, 3)), g])
        return g_padded, u

    def compute_ixn_grad_u(
        ff: Forcefield,
        precision,
        x0,
        box,
        lamb,
        num_water_atoms,
        host_system,
        water_idxs,
        ligand_idxs,
        protein_idxs,
        omm_topology,
        is_solvent=False,
    ):
        assert num_water_atoms == len(water_idxs)
        num_total_atoms = len(ligand_idxs) + len(protein_idxs) + num_water_atoms
        bt = Topology(ff)
        hgt = topology.HostGuestTopology(host_system.get_U_fns(), bt, num_water_atoms, ff, omm_topology)
        u = potentials.NonbondedInteractionGroup(
            num_total_atoms,
            ligand_idxs,
            hgt.host_nonbonded.potential.beta,
            hgt.host_nonbonded.potential.cutoff,
            col_atom_idxs=water_idxs if is_solvent else protein_idxs,
        )
        lig_params, _ = bt.parameterize_nonbonded(
            ff.q_handle.params,
            ff.q_handle_intra.params,
            ff.lj_handle.params,
            ff.lj_handle_intra.params,
            lamb=lamb,
            intramol_params=False,
        )
        host_ixn_params = hgt.host_nonbonded.params.copy()
        if not is_solvent and ff.env_bcc_handle is not None:  # protein
            env_bcc_h = ff.env_bcc_handle.get_env_handle(omm_topology, ff)
            host_ixn_params[:, NBParamIdx.Q_IDX] = env_bcc_h.parameterize(ff.env_bcc_handle.params)

        ixn_params = np.concatenate([host_ixn_params, lig_params])
        u_impl = u.bind(ixn_params).to_gpu(precision=precision).bound_impl
        return u_impl.execute(x0, box)

    with path_to_internal_file("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols_by_name = read_sdf_mols_by_name(path_to_ligand)

    # mol with no intramolecular NB terms and no dihedrals
    if use_tiny_mol:
        mol_h2s = Chem.AddHs(Chem.MolFromSmiles("S"))
        AllChem.EmbedMolecule(mol_h2s, randomSeed=2023)
        mols_by_name["H2S"] = mol_h2s

    if ctor == BaseTopology:
        if use_tiny_mol:
            mol = mols_by_name["H2S"]
        else:
            mol = mols_by_name["67"]
        ligand_conf = get_romol_conf(mol)
        Topology = partial(ctor, mol)
    elif ctor == DualTopology:
        if use_tiny_mol:
            mol_a = mols_by_name["H2S"]
            mol_b = mols_by_name["67"]
        else:
            # Pick smallest two molecules
            mol_a = mols_by_name["30"]
            mol_b = mols_by_name["67"]

        # Center mol to reduce overlap (high overlap fails in f32)
        mol_a_coords = get_romol_conf(mol_a)
        mol_a_center = np.mean(mol_a_coords, axis=0)
        mol_b_coords = get_romol_conf(mol_b)
        mol_b_center = np.mean(mol_b_coords, axis=0)
        mol_a_coords += mol_b_center - mol_a_center
        set_romol_conf(mol_a, mol_a_coords)

        ligand_conf = np.concatenate([get_romol_conf(mol_a), get_romol_conf(mol_b)])
        Topology = partial(ctor, mol_a, mol_b)
    else:
        raise ValueError(f"Unknown topology class: {ctor}")

    ligand_idxs = np.arange(ligand_conf.shape[0], dtype=np.int32)

    check_split_ixns(
        ligand_conf,
        ligand_idxs,
        precision,
        rtol,
        atol,
        compute_ref_grad_u,
        compute_new_grad_u,
        compute_intra_grad_u,
        compute_ixn_grad_u,
    )


@pytest.mark.nocuda
def test_exclude_all_ligand_ligand_ixns():
    num_host_atoms = 0
    num_guest_atoms = 3
    num_terms = num_guest_atoms * (num_guest_atoms - 1) // 2
    guest_exclusions, guest_scale_factors = topology.exclude_all_ligand_ligand_ixns(num_host_atoms, num_guest_atoms)
    assert (guest_exclusions == [[0, 1], [0, 2], [1, 2]]).all()
    assert (guest_scale_factors == np.ones((num_terms, 2))).all()

    num_host_atoms = 5
    num_guest_atoms = 3
    num_terms = num_guest_atoms * (num_guest_atoms - 1) // 2
    guest_exclusions, guest_scale_factors = topology.exclude_all_ligand_ligand_ixns(num_host_atoms, num_guest_atoms)
    assert (guest_exclusions == [[5, 6], [5, 7], [6, 7]]).all()
    assert (guest_scale_factors == np.ones((num_terms, 2))).all()

    num_host_atoms = 1
    num_guest_atoms = 5
    num_terms = num_guest_atoms * (num_guest_atoms - 1) // 2
    guest_exclusions, guest_scale_factors = topology.exclude_all_ligand_ligand_ixns(num_host_atoms, num_guest_atoms)
    assert (guest_exclusions == [[1, 2], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]]).all()
    assert (guest_scale_factors == np.ones((num_terms, 2))).all()

    num_host_atoms = 1
    num_guest_atoms = 0
    num_terms = num_guest_atoms * (num_guest_atoms - 1) // 2
    guest_exclusions, guest_scale_factors = topology.exclude_all_ligand_ligand_ixns(num_host_atoms, num_guest_atoms)
    assert guest_exclusions.shape == (0,)
    assert guest_scale_factors.shape == (0,)
