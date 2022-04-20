NUM_WORKERS = 12
import os

from timemachine.fe.topology import SingleTopologyV2

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(NUM_WORKERS)

import pymbar
from jax.config import config
from matplotlib import pyplot as plt
from rdkit.Chem import rdFMCS

from timemachine import constants

config.update("jax_enable_x64", True)
import functools

import jax
import numpy as np
from rdkit import Chem
from tqdm import tqdm

from timemachine import integrator
from timemachine.fe.utils import get_romol_conf, get_romol_masses
from timemachine.ff import Forcefield
from timemachine.ff.handlers.deserialize import deserialize_handlers
from timemachine.potentials import bonded


def make_conformer(mol_a, mol_b, conf_a, conf_b):
    mol = Chem.CombineMols(mol_a, mol_b)
    mol.RemoveAllConformers()  # necessary!
    cc = Chem.Conformer(mol.GetNumAtoms())
    conf = np.concatenate([conf_a, conf_b])
    conf *= 10  # TODO: label this unit conversion?
    for idx, pos in enumerate(np.asarray(conf)):
        cc.SetAtomPosition(idx, (float(pos[0]), float(pos[1]), float(pos[2])))
    mol.AddConformer(cc)

    return mol


import jax.numpy as jnp


def run_simulation(mol_a, mol_b, core):

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    print("core", core)
    ht = SingleTopologyV2(mol_a, mol_b, core, ff)

    src_bonds, dst_bonds = ht._parameterize_bonds(ff.hb_handle.params, ff.ha_handle.params, ff.pt_handle.params)

    # def parameterize(bp, ap, tp):
    #     src_bonds, dst_bonds = ht._parameterize_bonds(bp, ap, tp)
    #     hb_idxs_src, hb_params_src, ha_idxs_src, ha_params_src, pt_idxs_src, pt_params_src = src_bonds
    #     hb_idxs_dst, hb_params_dst, ha_idxs_dst, ha_params_dst, pt_idxs_dst, pt_params_dst = dst_bonds
    #     return (
    #         jnp.sum(jnp.array(hb_params_src))
    #         + jnp.sum(jnp.array(ha_params_src))
    #         + jnp.sum(jnp.array(pt_params_src))
    #         + jnp.sum(jnp.array(hb_params_dst))
    #         + jnp.sum(jnp.array(ha_params_dst))
    #         + jnp.sum(jnp.array(pt_params_dst))
    #     )
    # grad_fn = jax.grad(parameterize, argnums=0)
    # grads = grad_fn(ff.hb_handle.params, ff.ha_handle.params, ff.pt_handle.params)
    # print("grads", grads)
    # assert 0

    ht._draw_dummy_ixns()

    hb_idxs_src, hb_params_src, ha_idxs_src, ha_params_src, pt_idxs_src, pt_params_src = src_bonds
    hb_idxs_dst, hb_params_dst, ha_idxs_dst, ha_params_dst, pt_idxs_dst, pt_params_dst = dst_bonds

    def U_src(x):
        box = np.eye(3) * 100
        U_bond = bonded.harmonic_bond(
            x, params=np.array(hb_params_src), box=box, lamb=0.0, bond_idxs=np.array(hb_idxs_src)
        )
        U_angle = bonded.harmonic_angle(
            x, params=np.array(ha_params_src), box=box, lamb=0.0, angle_idxs=np.array(ha_idxs_src)
        )
        U_torsion = bonded.periodic_torsion(
            x, params=np.array(pt_params_src), box=box, lamb=0.0, torsion_idxs=np.array(pt_idxs_src)
        )
        return U_bond + U_angle + U_torsion

    def U_dst(x):
        box = np.eye(3) * 100
        U_bond = bonded.harmonic_bond(
            x, params=np.array(hb_params_dst), box=box, lamb=0.0, bond_idxs=np.array(hb_idxs_dst)
        )
        U_angle = bonded.harmonic_angle(
            x, params=np.array(ha_params_dst), box=box, lamb=0.0, angle_idxs=np.array(ha_idxs_dst)
        )
        U_torsion = bonded.periodic_torsion(
            x, params=np.array(pt_params_dst), box=box, lamb=0.0, torsion_idxs=np.array(pt_idxs_dst)
        )
        return U_bond + U_angle + U_torsion

    def U_combined(x, lamb):
        return (1 - lamb) * U_src(x) + lamb * U_dst(x)

    lambda_schedule = np.linspace(0.0, 1.0, 12)

    U_fns = []
    batch_U_fns = []

    for lamb in lambda_schedule:
        U_fn = functools.partial(U_combined, lamb=lamb)
        U_fns.append(jax.jit(U_fn))
        batch_U_fns.append(jax.vmap(U_fn))

    temperature = 300.0
    beta = 1 / (constants.BOLTZ * temperature)
    N_ks = []
    all_coords = []

    burn_in_batches = 20
    # num_batches = 1000  # total steps is 1000*NUM_WORKERS*steps_per_batch
    num_batches = 100  # total steps is 1000*NUM_WORKERS*steps_per_batch

    ligand_masses_a = get_romol_masses(mol_a)
    ligand_masses_b = get_romol_masses(mol_b)

    ligand_coords_a = get_romol_conf(mol_a)
    ligand_coords_b = get_romol_conf(mol_b)

    combined_masses = np.mean(ht.interpolate_params(ligand_masses_a, ligand_masses_b), axis=0)
    combined_coords = np.mean(ht.interpolate_params(ligand_coords_a, ligand_coords_b), axis=0)

    x0 = combined_coords

    print("Initial Energy", U_fn(x0))

    for idx, U_fn in enumerate(tqdm(U_fns)):

        coords = integrator.simulate(
            x0,
            U_fn,
            temperature,
            combined_masses,
            steps_per_batch=500,
            num_batches=num_batches + burn_in_batches,
            num_workers=NUM_WORKERS,
        )

        # toss away burn in batches and flatten
        coords = coords[:, burn_in_batches:, :, :].reshape(-1, x0.shape[0], 3)
        writer = Chem.SDWriter("out_" + str(idx) + ".sdf")
        all_coords.append(coords)
        N_ks.append(num_batches * NUM_WORKERS)

        for conf_c in coords:

            conf_a = np.zeros((mol_a.GetNumAtoms(), 3))
            conf_b = np.zeros((mol_b.GetNumAtoms(), 3))

            for a_idx, c_idx in enumerate(ht.a_to_c):
                conf_a[a_idx] = conf_c[c_idx]

            for b_idx, c_idx in enumerate(ht.b_to_c):
                conf_b[b_idx] = conf_c[c_idx]

            writer.write(make_conformer(mol_a, mol_b, conf_a, conf_b))
        writer.close()

    u_kns = []

    dG_estimate = 0
    dG_errs = []

    for idx in range(len(U_fns) - 1):

        fwd = batch_U_fns[idx + 1](all_coords[idx]) - batch_U_fns[idx](all_coords[idx])
        rev = batch_U_fns[idx](all_coords[idx + 1]) - batch_U_fns[idx + 1](all_coords[idx + 1])
        fwd *= beta
        rev *= beta

        # if idx == 0 or idx == len(U_fns) - 3 or idx == len(U_fns) - 2:
        # plt.hist(fwd, density=True, alpha=0.5, label="fwd")
        # plt.hist(-rev, density=True, alpha=0.5, label="-rev")
        # plt.legend()
        # plt.show()

        # print("fwd min/max", np.amin(fwd), np.amax(fwd))
        # print("rev min/max", np.amin(rev), np.amax(rev))
        # print("fwd nan count", np.sum(np.isnan(fwd)), "rev nan count", np.sum(np.isnan(rev)))

        dG, dG_err = pymbar.BAR(fwd, rev)
        dG_errs.append(dG_err)
        dG_estimate += dG

        print(idx, "->", idx + 1, dG / beta, dG_err / beta)  # in kJ

    dG_errs = np.array(dG_errs)

    all_coords = np.array(all_coords).reshape((-1, ht.NC, 3))
    for idx, U_batch in enumerate(batch_U_fns):
        reduced_nrg = U_batch(all_coords) * beta
        u_kns.append(reduced_nrg)

    u_kns = np.array(u_kns)
    N_ks = np.array(N_ks)

    obj = pymbar.MBAR(u_kns, N_k=N_ks)

    pbar_estimate = dG_estimate / beta
    pbar_err = np.linalg.norm(dG_errs) / beta
    mbar_estimate = (obj.f_k[-1] - obj.f_k[0]) / beta

    print(f"pair_bar {pbar_estimate:.3f} += {pbar_err:.3f} kJ/mol | mbar {mbar_estimate:.3f} kJ/mol")

    return dG_estimate / beta


class CompareDist(rdFMCS.MCSAtomCompare):
    def __init__(self, cutoff, *args, **kwargs):
        self.cutoff = cutoff * 10
        super().__init__(*args, **kwargs)

    def compare(self, p, mol1, atom1, mol2, atom2):
        x_i = mol1.GetConformer(0).GetPositions()[atom1]
        x_j = mol2.GetConformer(0).GetPositions()[atom2]
        if np.linalg.norm(x_i - x_j) > self.cutoff:
            return False
        else:
            return True


def get_core(mol_a: Chem.Mol, mol_b: Chem.Mol, cutoff: float = 0.1):

    ligand_coords_a = get_romol_conf(mol_a)
    ligand_coords_b = get_romol_conf(mol_b)

    mcs_params = rdFMCS.MCSParameters()
    mcs_params.AtomTyper = CompareDist(cutoff)
    mcs_params.BondCompareParameters.CompleteRingsOnly = 1
    mcs_params.BondCompareParameters.RingMatchesRingOnly = 1

    res = rdFMCS.FindMCS([mol_a, mol_b], mcs_params)

    query = Chem.MolFromSmarts(res.smartsString)

    mol_a_matches = mol_a.GetSubstructMatches(query, uniquify=False)
    mol_b_matches = mol_b.GetSubstructMatches(query, uniquify=False)

    best_match_dist = np.inf
    best_match_pairs = None
    for a_match in mol_a_matches:
        for b_match in mol_b_matches:
            dij = np.linalg.norm(ligand_coords_a[list(a_match)] - ligand_coords_b[list(b_match)])
            if dij < best_match_dist:
                best_match_dist = dij
                best_match_pairs = np.stack([a_match, b_match], axis=1)

    core_idxs = best_match_pairs

    assert len(core_idxs[:, 0]) == len(set(core_idxs[:, 0]))
    assert len(core_idxs[:, 1]) == len(set(core_idxs[:, 1]))

    return core_idxs


def test_hybrid_topology():

    suppl = Chem.SDMolSupplier("tests/data/ligands_40.sdf", removeHs=False)
    mols = [mol for mol in suppl]

    for i in range(len(mols)):
        for j in range(i + 1, len(mols)):

            mol_a = mols[i]
            mol_b = mols[j]

            if mol_a.GetProp("_Name") != "338" or mol_b.GetProp("_Name") != "67":
                continue

            print(
                "trying",
                mol_a.GetProp("_Name"),
                mol_b.GetProp("_Name"),
            )
            core = get_core(mol_a, mol_b)

            print("testing...", mol_a.GetProp("_Name"), mol_b.GetProp("_Name"), "core size", core.size)

            run_simulation(mol_a, mol_b, core)

            assert 0
