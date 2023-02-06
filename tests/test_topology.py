from functools import partial
from importlib import resources

import jax.numpy as jnp
import numpy as np
import pytest
from common import load_split_forcefields

from timemachine.fe import topology
from timemachine.fe.topology import BaseTopology, DualTopology, DualTopologyMinimization
from timemachine.fe.utils import get_mol_name, get_romol_conf, read_sdf
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import potentials
from timemachine.md.builders import build_water_system


def test_dual_topology_nonbonded_pairlist():
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        all_mols = read_sdf(path_to_ligand)

    mol_a = all_mols[1]
    mol_b = all_mols[4]
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    dt = topology.DualTopology(mol_a, mol_b, ff)

    nb_params, nb = dt.parameterize_nonbonded(ff.q_handle.params, ff.q_handle_intra.params, ff.lj_handle.params, 0.0)

    nb_pairlist_params, nb_pairlist = dt.parameterize_nonbonded_pairlist(
        ff.q_handle.params, ff.q_handle_intra.params, ff.lj_handle.params
    )

    x0 = np.concatenate([get_romol_conf(mol_a), get_romol_conf(mol_b)])
    box = np.eye(3) * 4.0

    for precision, rtol, atol in [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)]:

        nb_unbound = nb.unbound_impl(precision)
        nb_pairlist_unbound = nb_pairlist.unbound_impl(precision)

        du_dx, du_dp, u = nb_unbound.execute(x0, nb_params, box)

        pairlist_du_dx, pairlist_du_dp, pairlist_u = nb_pairlist_unbound.execute(x0, nb_pairlist_params, box)

        np.testing.assert_allclose(du_dx, pairlist_du_dx, atol=atol, rtol=rtol)

        # Different parameters, and so no expectation of shapes agreeing
        assert du_dp.shape != pairlist_du_dp.shape

        np.testing.assert_allclose(u, pairlist_u, atol=atol, rtol=rtol)


def parameterize_nonbonded_full(
    hgt: topology.HostGuestTopology, ff_q_params, ff_q_params_intra, ff_lj_params, lamb: float
):
    # Implements the full NB potential for the host guest system
    num_guest_atoms = hgt.guest_topology.get_num_atoms()
    guest_params, guest_pot = hgt.guest_topology.parameterize_nonbonded(
        ff_q_params, ff_q_params_intra, ff_lj_params, lamb
    )
    hg_exclusion_idxs = np.concatenate(
        [hgt.host_nonbonded.get_exclusion_idxs(), guest_pot.get_exclusion_idxs() + hgt.num_host_atoms]
    )
    hg_scale_factors = np.concatenate([hgt.host_nonbonded.get_scale_factors(), guest_pot.get_scale_factors()])
    hg_nb_params = jnp.concatenate([hgt.host_nonbonded.params, guest_params])
    return hg_nb_params, potentials.Nonbonded(
        hgt.num_host_atoms + num_guest_atoms,
        hg_exclusion_idxs,
        hg_scale_factors,
        guest_pot.get_beta(),
        guest_pot.get_cutoff(),
    )


@pytest.mark.parametrize("precision, rtol, atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
@pytest.mark.parametrize("ctor", [BaseTopology, DualTopology, DualTopologyMinimization])
def test_host_guest_nonbonded(ctor, precision, rtol, atol):
    def compute_ref_grad_u(ff: Forcefield, precision, x0, lamb):
        # Use the original code to compute the nb grads and potential
        bt = Topology(ff)
        hgt = topology.HostGuestTopology(host_bps, bt)
        params, potentials = parameterize_nonbonded_full(
            hgt, ff.q_handle.params, ff.q_handle_intra.params, ff.lj_handle.params, lamb=lamb
        )
        u_impl = potentials.bind(params).bound_impl(precision=precision)
        return u_impl.execute(x0, solvent_box)

    def compute_split_grad_u(ff: Forcefield, precision, x0, lamb):
        # Use the updated topology code to compute the nb grads and potential
        bt = Topology(ff)
        hgt = topology.HostGuestTopology(host_bps, bt)
        params, potentials = hgt.parameterize_nonbonded(
            ff.q_handle.params, ff.q_handle_intra.params, ff.lj_handle.params, lamb=lamb
        )
        u_impl = potentials.bind(params).bound_impl(precision=precision)
        return u_impl.execute(x0, solvent_box)

    def compute_vacuum_grad_u(ff: Forcefield, precision, x0, lamb):
        # Compute the vacuum nb grads and potential
        bt = Topology(ff)
        params, potentials = bt.parameterize_nonbonded(
            ff.q_handle.params, ff.q_handle_intra.params, ff.lj_handle.params, lamb=lamb
        )
        u_impl = potentials.bind(params).bound_impl(precision=precision)
        return u_impl.execute(x0, solvent_box)

    ffs = load_split_forcefields()

    box_width = 4.0
    solvent_sys, solvent_conf, solvent_box, solvent_top = build_water_system(box_width, ffs.ref.water_ff)
    solvent_box += np.diag([0.1, 0.1, 0.1])
    host_bps, host_masses = openmm_deserializer.deserialize_system(solvent_sys, cutoff=1.2)

    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols_by_name = {get_mol_name(mol): mol for mol in read_sdf(path_to_ligand)}

    if ctor == BaseTopology:
        mol = mols_by_name["43"]
        ligand_conf = get_romol_conf(mol)
        coords0 = np.concatenate([solvent_conf, ligand_conf])
        Topology = partial(ctor, mol)
    elif ctor in [DualTopology, DualTopologyMinimization]:
        mol_a = mols_by_name["43"]
        mol_b = mols_by_name["30"]
        ligand_conf = np.concatenate([get_romol_conf(mol_a), get_romol_conf(mol_b)])
        coords0 = np.concatenate([solvent_conf, ligand_conf])
        Topology = partial(ctor, mol_a, mol_b)
    else:
        raise ValueError(f"Unknown topology class: {ctor}")

    n_lambdas = 3
    for lamb in np.linspace(0, 1, n_lambdas):
        # Compute the grads, potential with the ref ff
        vacuum_grad_ref, vacuum_u_ref = compute_vacuum_grad_u(ffs.ref, precision, ligand_conf, lamb)
        solvent_grad_ref, solvent_u_ref = compute_ref_grad_u(ffs.ref, precision, coords0, lamb)

        # Compute the grads, potential with the scaled ff
        vacuum_grad_scaled, vacuum_u_scaled = compute_vacuum_grad_u(ffs.scaled, precision, ligand_conf, lamb)
        solvent_grad_scaled, solvent_u_scaled = compute_ref_grad_u(ffs.scaled, precision, coords0, lamb)

        # Compute the grads, potential with the intermol scaled ff
        vacuum_grad_inter_scaled, vacuum_u_inter_scaled = compute_vacuum_grad_u(
            ffs.inter_scaled, precision, ligand_conf, lamb
        )
        solvent_grad_inter_scaled, solvent_u_inter_scaled = compute_split_grad_u(
            ffs.inter_scaled, precision, coords0, lamb
        )

        # Compute the expected intermol scaled potential
        expected_inter_scaled_u = solvent_u_scaled - vacuum_u_scaled + vacuum_u_ref

        # Pad gradients for the solvent
        vacuum_grad_scaled_padded = np.concatenate([np.zeros(solvent_conf.shape), vacuum_grad_scaled])
        vacuum_grad_ref_padded = np.concatenate([np.zeros(solvent_conf.shape), vacuum_grad_ref])
        expected_inter_scaled_grad = solvent_grad_scaled - vacuum_grad_scaled_padded + vacuum_grad_ref_padded

        # They should be equal
        assert expected_inter_scaled_u == pytest.approx(solvent_u_inter_scaled, rel=rtol, abs=atol)
        np.testing.assert_allclose(expected_inter_scaled_grad, solvent_grad_inter_scaled, rtol=rtol, atol=atol)

        # The vacuum term should be the same as the ref
        assert vacuum_u_inter_scaled == pytest.approx(vacuum_u_ref, rel=rtol, abs=atol)
        np.testing.assert_allclose(vacuum_grad_ref, vacuum_grad_inter_scaled, rtol=rtol, atol=atol)
