import jax
import jax.numpy as jnp
import numpy as np
import os
import sys
import time

import simtk.unit
from rdkit import Chem
from rdkit.Chem import AllChem
from system import serialize
from system import forcefield

from openforcefield.typing.engines.smirnoff import ForceField
from simtk import openmm as mm
from simtk.openmm import app
from simtk.openmm.app import forcefield as ff

from scipy.stats import special_ortho_group
from jax.experimental import optimizers

import random
import multiprocessing

num_gpus = 8

from fe.common import minimize
from fe.utils import to_md_units, write

def rescale_and_center(conf, true_com, scale_factor=5):
    mol_com = np.sum(conf, axis=0)/conf.shape[0]
    # true_com = np.array([1.97698696, 1.90113478, 2.26042174]) # a-cd
    # true_com = np.array([0, 0, 0]) # water
    centered = conf - mol_com  # centered to origin
    return true_com + centered/scale_factor 


def initialize_off_parameters():
    '''
    Initializes openforcefield parameters for training.
    
    host_path (string): path to host if training binding energies (default = None)
    
    returns: (combined host and ligand parameters, smirnoff ligand parameters)
    '''
    # setting general smirnoff parameters for guest using random smiles string
    ref_mol = Chem.MolFromSmiles('CCCC')
    ref_mol = Chem.AddHs(ref_mol)
    AllChem.EmbedMolecule(ref_mol)

    smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")

    # smirnoff params are all encompassing for small molecules
    _, smirnoff_params, _, _, _ = forcefield.parameterize(ref_mol, smirnoff)
    return smirnoff_params
    # if host_path is not None:
    #     sys_xml = open(host_path, 'r').read()
    #     system = mm.XmlSerializer.deserialize(sys_xml)
    # else:
    #     system = host_sys
    # _, (host_params, _), _ = serialize.deserialize_system(system)       
    # epoch_combined_params = np.concatenate([host_params, smirnoff_params])

    # return epoch_combined_params


def run_simulation(args):
    mol, lamb, lambda_idx, smirnoff_params, pdb = args
    
    # print("rotating")
    # conf = mol.GetConformer(0)
    # coords = conf.GetPositions()
    # np.random.seed(int(time.time()+float(lambda_idx)))
    # rot_matrix = special_ortho_group.rvs(3).astype(dtype=np.float64)
    # coords = np.matmul(coords, rot_matrix)
    # for idx, (x,y,z) in enumerate(coords):
    #      conf.SetAtomPosition(idx, (x,y,z))

    p = multiprocessing.current_process()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(lambda_idx % num_gpus)

    # fname = "examples/water.pdb"
    omm_forcefield = app.ForceField('amber99sb.xml', 'tip3p.xml')
    # fname = "examples/BRD4/pdb/BRD4_10A_Belly.pdb"
    # pdb = app.PDBFile(fname)
    system = omm_forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False)
    
    coords = []
    for x,y,z in pdb.positions:
        coords.append([to_md_units(x),to_md_units(y),to_md_units(z)])
    coords = np.array(coords)
    host_conf = coords

    host_potentials, (host_params, host_param_groups), host_masses = serialize.deserialize_system(system)
    smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")

    guest_potentials, _, smirnoff_param_groups, guest_conf, guest_masses = forcefield.parameterize(mol, smirnoff)
    # guest_conf = rescale_and_center(guest_conf)

    host_conf = host_conf

    # combined s
    combined_potentials, combined_params, combined_param_groups, combined_conf, combined_masses = forcefield.combiner(
        guest_potentials, host_potentials,
        smirnoff_params, host_params,
        smirnoff_param_groups, host_param_groups,
        guest_conf, host_conf,
        guest_masses, host_masses)

    num_host_atoms = host_conf.shape[0]

    def filter_groups(param_groups, groups):
        roll = np.zeros_like(param_groups)
        for g in groups:
            roll = np.logical_or(roll, param_groups == g)
        return roll

    # host_dp_idxs = np.argwhere(filter_groups(host_param_groups, [7])).reshape(-1)
    # guest_dp_idxs = np.argwhere(filter_groups(smirnoff_param_groups, [7])).reshape(-1)

    # 1. host bond lengths
    # 7. host charges
    # 8. host vdw sigma
    # 9. host vdw epsilon
    # 19. ligand vdw epsilon

    combined_dp_idxs = np.argwhere(filter_groups(combined_param_groups, [17])).reshape(-1)

    print("Number of trainable parameters:", len(combined_dp_idxs))

    du_dls, du_dl_grads, all_xis = minimize(
        num_host_atoms,
        combined_potentials,
        combined_params,
        combined_conf,
        combined_masses,
        combined_dp_idxs,
        starting_dimension=4,
        lamb=lamb
    )


    # debug: write out the trajectory.

    coords_buffer = []
    for xi in all_xis:
        coords_buffer.append(write(np.asarray(xi[:, :3]*10), combined_masses))
    coords_buffer = "".join(coords_buffer)

    return lamb, du_dls, du_dl_grads, coords_buffer, combined_dp_idxs

def run_leg(lambda_schedule, epoch_params, mol, pdb, prefix):
    """
    Run a leg of the simulation.

    Parameters
    ----------
    lambda_schedule: list of float
        Lambda schedule of the leg

    epoch_params: current forcefield parameters
        Epoch params, currently corresponding to the small molecule forcefield

    mol: rdkit mol
        Small molecule of interest

    pdb: openmm.PDBFile
        PDB file of the environment

    prefix: string
        For debugging use

    Returns
    -------
    delta_G, delta_G grads
        delta G of the leg and its gradient

    """
    # lambda_schedule = [0.15, 0.25]
    pool = multiprocessing.Pool(num_gpus)

    args = []
    for lamb_idx, lamb in enumerate(lambda_schedule):
        params = (mol, lamb, lamb_idx, epoch_params, pdb)
        args.append(params)

    results = pool.map(run_simulation, args)

    all_lambdas = []
    all_mean_du_dls = []
    all_mean_du_dl_grads = []

    for lamb_idx, (lamb, du_dls, du_dl_grads, coords_buffer, combined_dp_idxs) in enumerate(results):
        all_lambdas.append(lamb)
        all_mean_du_dls.append(np.mean(du_dls))
        all_mean_du_dl_grads.append(np.mean(du_dl_grads, axis=0))
        with open(prefix+"_"+str(lamb_idx)+"_coords.xyz", "w") as fh:
            fh.write(coords_buffer)

    all_lambdas = np.array(all_lambdas) 
    all_mean_du_dls = np.array(all_mean_du_dls)
    all_mean_du_dl_grads = np.array(all_mean_du_dl_grads)


    num_params = all_mean_du_dl_grads.shape[-1]

    print(prefix,"du_dls",all_mean_du_dls)
    print(prefix,"lambdas",all_lambdas)

    pred_dG = np.trapz(all_mean_du_dls, all_lambdas)

    dG_grads = []
    dG_grads_median = []

    for p_idx in range(num_params):
        dG_grads.append(np.trapz(all_mean_du_dl_grads[:, p_idx], all_lambdas))

    dG_grads = np.array(dG_grads)

    dparams = np.zeros_like(epoch_params)
    dparams[combined_dp_idxs] = dG_grads

    pool.close()

    return pred_dG, dparams


def train(true_dG):
    fname = "examples/BRD4/mol2/ligand-4.mol2"
    guest_mol2 = open(fname, "r").read()
    mol = Chem.MolFromMol2Block(guest_mol2, sanitize=True, removeHs=False, cleanupSubstructures=True)

    # guest_mol2 = Chem.MolFromSmiles("c1ccccc1")
    # mol = Chem.AddHs(guest_mol2)
    # mol = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol), removeHs=False)
    # print(mol)

    # AllChem.EmbedMolecule(mol, randomSeed=1337)
    # AllChem.EmbedMolecule(mol)

    omm_forcefield = app.ForceField('amber99sb.xml', 'tip3p.xml') # for proteins

    # both of these need to be centered around the ligand
    protein_ligand_pdb = app.PDBFile('examples/BRD4/pdb/BRD4_10A_Belly.pdb')
    water_pdb = app.PDBFile('examples/BRD4/water_shell.pdb')

    starting_off_params = initialize_off_parameters()
    # assert 0
    lr = 5e-3
    opt_init, opt_update, get_params = optimizers.adam(lr)
    # opt_init, opt_update, get_params = optimizers.sgd(lr)

    opt_state = opt_init(starting_off_params)

    num_epochs = 50
    for epoch in range(num_epochs):

        # print("turning off special ortho")
        # print("WARNING: turning on special ortho")
        # conf = mol.GetConformer(0)
        # coords = conf.GetPositions()
        # rot_matrix = special_ortho_group.rvs(3).astype(dtype=np.float64)
        # coords = np.matmul(coords, rot_matrix)
        # for idx, (x,y,z) in enumerate(coords):
        #    conf.SetAtomPosition(idx, (x,y,z))

        print("===============Epoch "+str(epoch)+"=============")

        solvent_params = []
        complex_params = []
        all_lambdas = []
        # lambda_schedule = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 4.0, 6.0, 8.0, 10.0]
        # lambda_schedule = [0.0, 0.05, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0, 1.5,2.5,3.5,5.0,10.0,250.0]
        # lambda_schedule = [0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 5.0, 10.0]
        # lambda_schedule = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
        # lambda_schedule = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        # lambda_schedule = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        # lambda_schedule = [0.0, 25.0, 250.0, 2500.0, 100000.0]
        # lambda_schedule = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # lambda_schedule = [0.0]
        lambda_schedule = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.5, 10.0]

        # lambda_schedule = [0.15, 0.25]

        epoch_off_params = get_params(opt_state)

        complex_dG, complex_dG_grads = run_leg(lambda_schedule, epoch_off_params, mol, protein_ligand_pdb, "complex_e"+str(epoch))
        solvent_dG, solvent_dG_grads = run_leg(lambda_schedule, epoch_off_params, mol, water_pdb, "solvent_e"+str(epoch))





        # print("saving ff params")
        # np.savez("epoch_run_solvation"+str(epoch), params=epoch_off_params)

        # for lamb_idx, lamb in enumerate(lambda_schedule):
        #     s_params = (mol, lamb, lamb_idx, epoch_off_params, water_pdb)
        #     solvent_params.append(s_params)
        #     c_params = (mol, lamb, lamb_idx, epoch_off_params, protein_ligand_pdb)
        #     complex_params.append(c_params)

        # # run_simulation(solvent_params[0])
        # # assert 0

        # print("???")
        # results = pool.map(run_simulation, solvent_params)

        # all_lambdas = []
        # all_mean_du_dls = []
        # all_mean_du_dl_grads = []
        # all_energies = []
        # for lamb, du_dls, du_dl_grads, nrgs, combined_dp_idxs in results:
        #     all_lambdas.append(lamb)
        #     all_mean_du_dls.append(np.mean(du_dls))
        #     all_mean_du_dl_grads.append(np.mean(du_dl_grads, axis=0))
        #     all_energies.append(nrgs)

        # all_energies = np.array(all_energies)
        # all_lambdas = np.array(all_lambdas) 
        # all_mean_du_dls = np.array(all_mean_du_dls)
        # all_mean_du_dl_grads = np.array(all_mean_du_dl_grads)

        # num_params = all_mean_du_dl_grads.shape[-1]

        # # np.set_printoptions(linewidth=np.inf)
    
        # # print("MEAN DU_DL", all_mean_du_dls)

        # # for p_idx in range(num_params):
        # #     print("MEAN DU_DL_GRAD_"+str(p_idx), all_mean_du_dl_grads[:, p_idx])

        # pred_dG = np.trapz(all_mean_du_dls, all_lambdas)

        # dG_grads = []
        # dG_grads_median = []

        # for p_idx in range(num_params):
        #     dG_grads.append(np.trapz(all_mean_du_dl_grads[:, p_idx], all_lambdas))

        # dG_grads = np.array(dG_grads)

        # dparams = np.zeros_like(epoch_params)
        # dparams[combined_dp_idxs] = dG_grads

        pred_dG = complex_dG - solvent_dG
        pred_dG_grad = complex_dG_grads - solvent_dG_grads

        L2_loss = (pred_dG-true_dG)**2
        # print(pred_dG-true_dG)
        # print(pred_dG_grad)
        L2_grad = 2*(pred_dG-true_dG)*pred_dG_grad

        print("dG_grad max/min", np.amax(pred_dG_grad), np.amin(pred_dG_grad))

        L1_loss = np.abs(pred_dG-true_dG)
        print("l1 loss", L1_loss, "\t kJ/mol", "true vs pred", true_dG, "\t", pred_dG)

        # fix me when going to multiple molecules
        # print("full_L2_grad", full_L2_grad)
        opt_state = opt_update(epoch, L2_grad, opt_state)

    # pool.close()

train(3.575*4.18)
