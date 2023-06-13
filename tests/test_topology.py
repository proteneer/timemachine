from functools import partial
from importlib import resources

import jax.numpy as jnp
import numpy as np
import pytest
from common import check_split_ixns
from rdkit import Chem
from rdkit.Chem import AllChem

from timemachine import potentials
from timemachine.fe import topology
from timemachine.fe.topology import BaseTopology, DualTopology, DualTopologyMinimization
from timemachine.fe.utils import get_mol_name, get_romol_conf, read_sdf, set_romol_conf
from timemachine.ff import Forcefield


def test_dual_topology_nonbonded_pairlist():
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        all_mols = read_sdf(path_to_ligand)

    mol_a = all_mols[1]
    mol_b = all_mols[4]
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    dt = topology.DualTopology(mol_a, mol_b, ff)

    nb_params, nb = dt.parameterize_nonbonded(
        ff.q_handle.params, ff.q_handle_intra.params, ff.q_handle_solv.params, ff.lj_handle.params, 0.0
    )

    nb_pairlist_params, nb_pairlist = dt.parameterize_nonbonded_pairlist(
        ff.q_handle.params, ff.q_handle_intra.params, ff.lj_handle.params
    )

    x0 = np.concatenate([get_romol_conf(mol_a), get_romol_conf(mol_b)])
    box = np.eye(3) * 4.0

    for precision, rtol, atol in [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)]:

        nb_unbound = nb.to_gpu(precision).unbound_impl
        nb_pairlist_unbound = nb_pairlist.to_gpu(precision).unbound_impl

        du_dx, du_dp, u = nb_unbound.execute(x0, nb_params, box)

        pairlist_du_dx, pairlist_du_dp, pairlist_u = nb_pairlist_unbound.execute(x0, nb_pairlist_params, box)

        np.testing.assert_allclose(du_dx, pairlist_du_dx, atol=atol, rtol=rtol)

        # Different parameters, and so no expectation of shapes agreeing
        assert du_dp.shape != pairlist_du_dp.shape

        np.testing.assert_allclose(u, pairlist_u, atol=atol, rtol=rtol)


def parameterize_nonbonded_full(
    hgt: topology.HostGuestTopology, ff_q_params, ff_q_params_intra, ff_q_params_solv, ff_lj_params, lamb: float
):
    # Implements the full NB potential for the host guest system
    num_guest_atoms = hgt.guest_topology.get_num_atoms()
    guest_params, guest_pot = hgt.guest_topology.parameterize_nonbonded(
        ff_q_params, ff_q_params_intra, ff_q_params_solv, ff_lj_params, lamb
    )
    hg_exclusion_idxs = np.concatenate(
        [hgt.host_nonbonded.potential.exclusion_idxs, guest_pot.exclusion_idxs + hgt.num_host_atoms]
    )
    hg_scale_factors = np.concatenate([hgt.host_nonbonded.potential.scale_factors, guest_pot.scale_factors])
    hg_nb_params = jnp.concatenate([hgt.host_nonbonded.params, guest_params])
    return hg_nb_params, potentials.Nonbonded(
        hgt.num_host_atoms + num_guest_atoms, hg_exclusion_idxs, hg_scale_factors, guest_pot.beta, guest_pot.cutoff
    )


@pytest.mark.parametrize("precision, rtol, atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
@pytest.mark.parametrize("ctor", [BaseTopology, DualTopology, DualTopologyMinimization])
@pytest.mark.parametrize("use_tiny_mol", [True, False])
def test_host_guest_nonbonded(ctor, precision, rtol, atol, use_tiny_mol):
    def compute_ref_grad_u(ff: Forcefield, precision, x0, box, lamb, num_water_atoms, host_bps):
        # Use the original code to compute the nb grads and potential
        bt = Topology(ff)
        hgt = topology.HostGuestTopology(host_bps, bt, num_water_atoms)
        params, us = parameterize_nonbonded_full(
            hgt, ff.q_handle.params, ff.q_handle_intra.params, ff.q_handle_solv.params, ff.lj_handle.params, lamb=lamb
        )
        u_impl = us.bind(params).to_gpu(precision=precision).bound_impl
        return u_impl.execute(x0, box)

    def compute_new_grad_u(ff: Forcefield, precision, x0, box, lamb, num_water_atoms, host_bps):
        # Use the updated topology code to compute the nb grads and potential
        bt = Topology(ff)
        hgt = topology.HostGuestTopology(host_bps, bt, num_water_atoms)
        params, us = hgt.parameterize_nonbonded(
            ff.q_handle.params,
            ff.q_handle_intra.params,
            ff.q_handle_solv.params,
            ff.lj_handle.params,
            lamb=lamb,
        )
        u_impl = us.bind(params).to_gpu(precision=precision).bound_impl
        return u_impl.execute(x0, box)

    def compute_intra_grad_u(ff: Forcefield, precision, x0, box, lamb, num_water_atoms, num_host_atoms):
        # Compute the vacuum nb grads and potential for the ligand intramolecular term
        bt = Topology(ff)
        params, us = bt.parameterize_nonbonded(
            ff.q_handle.params, ff.q_handle_intra.params, ff.q_handle_solv.params, ff.lj_handle.params, lamb=lamb
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
        host_bps,
        water_idxs,
        ligand_idxs,
        protein_idxs,
        is_solvent=False,
    ):
        assert num_water_atoms == len(water_idxs)
        num_total_atoms = len(ligand_idxs) + len(protein_idxs) + num_water_atoms
        bt = Topology(ff)
        hgt = topology.HostGuestTopology(host_bps, bt, num_water_atoms)
        u = potentials.NonbondedInteractionGroup(
            num_total_atoms,
            ligand_idxs,
            hgt.host_nonbonded.potential.beta,
            hgt.host_nonbonded.potential.cutoff,
            col_atom_idxs=water_idxs if is_solvent else protein_idxs,
        )
        lig_params, _ = bt.parameterize_nonbonded(
            ff.q_handle_solv.params if is_solvent else ff.q_handle.params,
            ff.q_handle_intra.params,
            ff.q_handle_solv.params,
            ff.lj_handle.params,
            lamb=lamb,
            intramol_params=False,
        )
        ixn_params = np.concatenate([hgt.host_nonbonded.params, lig_params])
        u_impl = u.bind(ixn_params).to_gpu(precision=precision).bound_impl
        return u_impl.execute(x0, box)

    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols_by_name = {get_mol_name(mol): mol for mol in read_sdf(path_to_ligand)}

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
    elif ctor in [DualTopology, DualTopologyMinimization]:
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
