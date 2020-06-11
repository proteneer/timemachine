import copy
import argparse
import time
import numpy as np
from io import StringIO
import itertools
import os
import sys

from timemachine.integrator import langevin_coefficients

from jax.config import config as jax_config
# this always needs to be set
jax_config.update("jax_enable_x64", True)

from scipy.stats import special_ortho_group
import jax
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from simtk.openmm import app
from simtk.openmm.app import PDBFile

from timemachine.lib import custom_ops, ops
from timemachine.potentials import jax_utils
from fe.utils import to_md_units, write
from fe import math_utils

from multiprocessing import Process, Pipe
from matplotlib import pyplot as plt

from jax.experimental import optimizers

from fe import simulation_v2
from fe import loss, bar
from fe.pdb_writer import PDBWriter

from ff import forcefield
from ff import system
from ff import openmm_converter
import jax.numpy as jnp

from rdkit.Chem import rdFMCS


def convert_uIC50_to_kJ_per_mole(amount_in_uM):
    return 0.593*np.log(amount_in_uM*1e-6)*4.18


def loss_fn(all_du_dls, true_ddG, lambda_schedule):
    """
    Loss function. Currently set to L1.

    Parameters:
    -----------
    all_du_dls: shape [L, F, T] np.array
        Where L is the number of lambda windows, F is the number of forces, and T is the total number of equilibrated steps

    true_ddG: scalar
        True ddG of the edge.

    Returns:
    --------
    scalar
        Loss

    """
    assert all_du_dls.ndim == 3

    total_du_dls = jnp.sum(all_du_dls, axis=1) # shape [L, T]
    mean_du_dls = jnp.mean(total_du_dls, axis=1) # shape [L]
    pred_ddG = math_utils.trapz(mean_du_dls, lambda_schedule)
    return jnp.abs(pred_ddG - true_ddG)

loss_fn_grad = jax.grad(loss_fn, argnums=(0,))

def setup_core_restraints(
    k,
    count,
    conf,
    nha,
    core_atoms,
    params,
    nrg_fns,
    stage):
    """
    Setup core restraints

    Parameters
    ----------
    k: float
        Force constant of each restraint

    count: int
        Number of host atoms we restrain each ligand to

    nha: int
        Number of host atoms

    core_atoms: list of int
        atoms we're restraining. This is indexed by the total number of atoms in the system.

    params: np.array float
        fundamental parameters we're modifying

    nrg_fns: dict
        nrg fns from the timemachine System object.

    stage: 0,1,2
        0 - attach restraint
        1 - decouple
        2 - detach restraint

    """
    ri = np.expand_dims(conf, axis=0)
    rj = np.expand_dims(conf, axis=1)
    dij = jax_utils.distance(ri, rj)
    all_nbs = []

    bond_param_idxs = []
    bond_idxs = []

    for l_idx, dists in enumerate(dij[nha:]):
        if l_idx in core_atoms:
            nns = np.argsort(dists[:nha])

            # restrain to 10 nearby atoms
            for p_idx in nns[:10]:
                k_idx = len(params)
                params = np.concatenate([params, [k]])

                b = dists[p_idx]
                b_idx = len(params)
                params = np.concatenate([params, [b]])

                bond_param_idxs.append([k_idx, b_idx])
                bond_idxs.append([l_idx + nha, p_idx])

    bond_idxs = np.array(bond_idxs, dtype=np.int32)
    bond_param_idxs = np.array(bond_param_idxs, dtype=np.int32)

    B = bond_idxs.shape[0]

    if stage == 0:
        lambda_flags = np.ones(B, dtype=np.int32)
    elif stage == 1:
        lambda_flags = np.zeros(B, dtype=np.int32)
    elif stage == 2:
        lambda_flags = np.ones(B, dtype=np.int32)

    nrg_fns['FlatBottom'] = (
        bond_idxs,
        bond_param_idxs,
        lambda_flags,
        0
    )

    return params


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Relative Binding Free Energy Script')
    parser.add_argument('--out_dir', type=str, required=True, help='Location of all output files')
    parser.add_argument('--precision', type=str, required=True, help='Either single or double precision. Double is 8x slower.')
    parser.add_argument('--protein_pdb', type=str, required=True, help='Prepared protein PDB file. This should not have any waters.')
    parser.add_argument('--ligand_sdf', type=str, required=True, help='The ligand sdf used along with posed 3D coordinates. Only the first two ligands are used.')
    parser.add_argument('--num_gpus', type=int, required=True, help='Number of gpus available.')
    parser.add_argument('--forcefield', type=str, required=True, help='Small molecule forcefield to be loaded.')
    parser.add_argument('--seed', type=int, required=True, help='Random seed used for all the random number generators.')
    parser.add_argument('--cutoff', type=float, required=True, help='Nonbonded cutoff. Please set this to 1.0 for now.')
    parser.add_argument('--lamb', type=float, required=False, help='Which lambda window we run at.')
    parser.add_argument('--n_frames', type=int, required=True, help='Number of PDB frames to write. If 0 then writing is skipped entirely.')
    parser.add_argument('--steps', type=int, required=True, help='Number of steps we run')
    parser.add_argument('--a_idx', type=int, required=True, help='A index')
    parser.add_argument('--restr_force', type=float, required=True, help='Strength of the each restraint term, in kJ/mol.')
    parser.add_argument('--restr_count', type=int, required=True, help='Number of host atoms we restrain each core atom to.')

    args = parser.parse_args()

    assert os.path.isdir(args.out_dir)

    if args.precision == 'single':
        precision = np.float32
    elif args.precision == 'double':
        precision = np.float64
    else:
        raise Exception("precision must be either single or double")

    suppl = Chem.SDMolSupplier(args.ligand_sdf, removeHs=False)

    all_guest_mols = []
    for guest_idx, guest_mol in enumerate(suppl):
        all_guest_mols.append(guest_mol)

    mol_a = all_guest_mols[args.a_idx]

    mol_a_dG = convert_uIC50_to_kJ_per_mole(float(mol_a.GetProp("IC50[uM](SPA)")))

    a_name = mol_a.GetProp("_Name")

    print("Ligand A Name:", a_name)

    open_ff = forcefield.Forcefield(args.forcefield)
    all_nrg_fns = []

    # a_system = open_ff.parameterize(mol_a, cutoff=args.cutoff, am1=True, zero_charges=True)
    a_system = open_ff.parameterize(mol_a, am1=True, zero_charges=False)

    host_pdb_file = args.protein_pdb
    host_pdb = app.PDBFile(host_pdb_file)
    amber_ff = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
    host_system = amber_ff.createSystem(
        host_pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False
    )

    lambda_plane_idxs = np.zeros(mol_a.GetNumAtoms())

    host_conf = []
    for x,y,z in host_pdb.positions:
        host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])
    host_conf = np.array(host_conf)

    conformer = mol_a.GetConformer(0)
    mol_a_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    mol_a_conf = mol_a_conf/10 # convert to md_units

    x0 = np.concatenate([host_conf, mol_a_conf]) # combined geometry

    host_system = openmm_converter.deserialize_system(host_system)

    # (use this if you want to manually set the core)
    # print("Warning: manually setting core atoms, use the geometric MCS script later to deal with this.")
    # core_atoms = [4,5,6,7,8,9,10,11,12,13,15,16,18], specific to mol name 338
    core_smarts = '[#6]1:[#6]:[#6]:[#6](:[#6](:[#6]:1-[#8]-[#6](:[#6]-[#1]):[#6])-[#1])-[#1]'

    print("Using core smarts:", core_smarts)
    core_query = Chem.MolFromSmarts(core_smarts)
    core_atoms = mol_a.GetSubstructMatch(core_query)

    print("Core atoms to be restrained:", core_atoms)

    for epoch in range(100):

        print("Starting epoch -----"+str(epoch)+'-----')

        stage_dGs = []
    
        epoch_dir = os.path.join(args.out_dir, "epoch_"+str(epoch))

        for stage in [0, 1, 2]:

            print("---Starting stage", stage, '---')

            stage_dir = os.path.join(epoch_dir, "stage_"+str(stage))

            if not os.path.exists(stage_dir):
                os.makedirs(stage_dir)

            if args.lamb:
                ti_lambdas = np.ones(args.num_gpus)*args.lamb
            else:
                if stage == 0 or stage == 2:
                    # lambda spans from [0, 1], and are analytically zero at endpoints
                    ti_lambdas = np.linspace(0.0, 1.0, 16)
                elif stage == 1:
                    # lambda spans from [0, inf], is close enough to zero over [0, 1.2] cutoff
                    ti_lambdas = np.concatenate([
                        np.linspace(0.0, 0.5, 24, endpoint=False),
                        np.linspace(0.5, 1.2, 8)
                    ])
                else:
                    raise Exception("Unknown stage.")

            combined_system = host_system.merge(
                a_system,
                stage=stage,
                nonbonded_cutoff=1000.0
            )

            nha = host_conf.shape[0]
            new_params = setup_core_restraints(
                args.restr_force,
                args.restr_count,
                x0,
                nha,
                core_atoms,
                combined_system.params,
                combined_system.nrg_fns,
                stage=stage
            )

            temperature = 300
            dt = 1.5e-3
            friction = 40

            masses = np.array(combined_system.masses)
            ca, cbs, ccs = langevin_coefficients(
                temperature,
                dt,
                friction,
                masses
            )

            cbs *= -1

            print("Integrator coefficients:")
            print("ca", ca)
            print("cbs", cbs)
            print("ccs", ccs)

            complete_T = args.steps

            equil_T = 2000

            # we use only samples after this number of steps when computing dGs
            du_dl_cutoff = 20000

            assert complete_T > equil_T
            assert complete_T > du_dl_cutoff
            
            all_processes = []
            all_pcs = []

            for lambda_idx, lamb in enumerate(ti_lambdas):

                complete_lambda = np.zeros(complete_T) + lamb
                complete_cas = np.ones(complete_T)*ca
                complete_dts = np.concatenate([
                    np.linspace(0, dt, equil_T),
                    np.ones(complete_T-equil_T)*dt
                ])

                sim = simulation_v2.Simulation(
                    combined_system,
                    complete_dts,
                    complete_cas,
                    cbs,
                    ccs,
                    complete_lambda,
                    precision
                )

                intg_seed = np.random.randint(np.iinfo(np.int32).max)
                # intg_seed = 2020

                combined_pdb = Chem.CombineMols(Chem.MolFromPDBFile(host_pdb_file, removeHs=False), mol_a)
                combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))
                out_file = os.path.join(stage_dir, str(epoch)+"_abfe_"+str(lambda_idx)+".pdb")
                writer = PDBWriter(combined_pdb_str, out_file, args.n_frames)

                # zero-out
                # if args.n_frames is 0:
                    # writer = None
                # writer = None

                v0 = np.zeros_like(x0)

                parent_conn, child_conn = Pipe()

                input_args = (x0, v0, new_params, intg_seed, writer, child_conn, lambda_idx % args.num_gpus, du_dl_cutoff)
                p = Process(target=sim.run_forward_and_backward, args=input_args)

                # sim.run_forward_and_backward(*input_args)

                all_pcs.append(parent_conn)
                all_processes.append(p)

            sum_du_dls = [] # [L, T]
            all_du_dls = [] # [L, F, T] num lambda windows, num forces, num steps

            all_energies = []

            # run inference loop to generate all_du_dls
            for b_idx in range(0, len(all_processes), args.num_gpus):
                for p in all_processes[b_idx:b_idx+args.num_gpus]:
                    p.start()

                batch_du_dls = []
                for pc_idx, pc in enumerate(all_pcs[b_idx:b_idx+args.num_gpus]):

                    lamb_idx = b_idx+pc_idx
                    lamb = ti_lambdas[b_idx+pc_idx]

                    offset = equil_T # TBD use this
                    full_du_dls, full_energies = pc.recv() # (F, T), (T)
                    pc.send(None)
                    assert full_du_dls is not None
                    total_du_dls = np.sum(full_du_dls, axis=0)

                    plt.plot(total_du_dls, label="{:.2f}".format(lamb))
                    plt.ylabel("du_dl")
                    plt.xlabel("timestep")
                    plt.legend()
                    fpath = os.path.join(stage_dir, "lambda_du_dls_"+str(lamb_idx))
                    plt.savefig(fpath)
                    plt.clf()

                    plt.plot(full_energies, label="{:.2f}".format(lamb))
                    plt.ylabel("U")
                    plt.xlabel("timestep")
                    plt.legend()

                    fpath = os.path.join(stage_dir, "lambda_energies_"+str(lamb_idx))
                    plt.savefig(fpath)
                    plt.clf()

                    sum_du_dls.append(total_du_dls)
                    all_du_dls.append(full_du_dls)

            # compute loss and derivatives w.r.t. adjoints
            all_du_dls = np.array(all_du_dls)
            sum_du_dls = np.array(sum_du_dls)

            stage_dG = np.trapz(np.mean(sum_du_dls[:, du_dl_cutoff:], axis=1), ti_lambdas)
            print("stage", stage, "pred_dG", stage_dG)

            plt.clf()
            plt.violinplot(sum_du_dls[:, du_dl_cutoff:].tolist(), positions=ti_lambdas)
            plt.ylabel("du_dlambda")
            plt.savefig(os.path.join(stage_dir, "violin_du_dls"))
            plt.clf()

            plt.boxplot(sum_du_dls[:, du_dl_cutoff:].tolist(), positions=ti_lambdas)
            plt.ylabel("du_dlambda")
            plt.savefig(os.path.join(stage_dir, "boxplot_du_dls"))
            plt.clf()
            
            stage_dGs.append(stage_dG)

        print("epoch", epoch, "stage_dGs", stage_dGs, "final dG", stage_dGs[0]+stage_dGs[1]-stage_dGs[2])
