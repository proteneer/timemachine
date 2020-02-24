
from jax.config import config as jax_config
# this always needs to be set
jax_config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import numpy as np
import os
import sys
import time
import itertools
import argparse
from io import StringIO

import simtk.unit
from rdkit import Chem
from rdkit.Chem import AllChem
from system import serialize
from system import forcefield

from openforcefield.typing.engines.smirnoff import ForceField
from simtk import openmm as mm
from simtk.openmm import app
from simtk.openmm.app import forcefield as ff
from simtk.openmm.app import PDBFile

from scipy.stats import special_ortho_group
from jax.experimental import optimizers

import random
import multiprocessing

from fe.common import minimize, convert_uIC50_to_kJ_per_mole
from fe.utils import to_md_units, write
from fe.dataset import Dataset


from system import custom_functionals

# why does this run in the same time
custom_functionals.set_double_precision()
# custom_functionals.set_single_precision()


def rescale_and_center(conf, true_com=None, scale_factor=2):
    mol_com = np.sum(conf, axis=0)/conf.shape[0]
    centered = conf - mol_com  # centered to origin
    if true_com is None:
        true_com = mol_com
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

    am1 = False
    assert am1 is False

    # smirnoff params are all encompassing for small molecules

    _, smirnoff_params, _, _, _ = forcefield.parameterize(ref_mol, smirnoff, am1)
    return smirnoff_params


class PDBWriter():

    def __init__(self, pdb_str, outfile):
        cpdb = app.PDBFile(pdb_str)
        PDBFile.writeHeader(cpdb.topology, outfile)
        self.topology = cpdb.topology
        self.frame_idx = 0
        self.outfile = outfile

    def write(self, x):
        self.frame_idx += 1
        PDBFile.writeModel(self.topology, x, self.outfile, self.frame_idx)

    def close(self):
        PDBFile.writeFooter(self.topology, self.outfile)
        self.outfile.flush()



def run_simulation(mol_name, mol, smirnoff_params, pdb, inference):
    omm_forcefield = app.ForceField('amber99sb.xml', 'tip3p.xml')
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

    # CAREFUL THIS SHOULD BE THE SINGLE SET OF CONSISTENT FORCEFIELDS
    am1 = False
    if am1:
        guest_potentials, smirnoff_params, smirnoff_param_groups, guest_conf, guest_masses = forcefield.parameterize(mol, smirnoff, am1)
    else:
        guest_potentials, _, smirnoff_param_groups, guest_conf, guest_masses = forcefield.parameterize(mol, smirnoff, am1)

    # guest_conf = rescale_and_center(guest_conf, scale_factor=3.0)

    combined_potentials, combined_params, combined_param_groups, combined_conf, combined_masses = forcefield.combiner(
        host_potentials, guest_potentials, 
        host_params, smirnoff_params, 
        host_param_groups, smirnoff_param_groups, 
        host_conf, guest_conf, 
        host_masses, guest_masses)

    num_host_atoms = host_conf.shape[0]

    print("Combined conf shape", combined_conf.shape)

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

    # 17. charge

    combined_dp_idxs = np.argwhere(filter_groups(combined_param_groups, [17])).reshape(-1)
    if inference:
        combined_dp_idxs = []

    print("CDPI shape", combined_dp_idxs.shape)

    # proper truncation
    # combined_dp_idxs = combined_dp_idxs[:13]

    outfile = open("frames/"+mol_name+".pdb", "w")

    combined_pdb = Chem.CombineMols(Chem.MolFromPDBFile(pdb.filepath, removeHs=False), mol)
    combined_pdb_str = StringIO(Chem.MolToPDBBlock(combined_pdb))

    writer = PDBWriter(combined_pdb_str, outfile)

    dG_insertion, dG_deletion = minimize(
        mol_name,
        num_host_atoms,
        combined_potentials,
        combined_params,
        combined_conf,
        combined_masses,
        combined_dp_idxs,
        writer_fn=writer.write
    )

    writer.close()
    # assert 0

    # # PDBFile.writeFooter(cpdb.topology, outfile)
    # # outfile.flush()

    # # print("CDPI", combined_dp_idxs)
    # # (ytz): we need to offset this by number of host params to compute the indices
    # # of the original ligand parameters.
    # if not inference:
    #     ligand_dp_idxs = combined_dp_idxs - len(host_params)
    # else:
    #     ligand_dp_idxs = None

    # print("dG", dG)

    # return dG, dG_grads, ligand_dp_idxs
    print("insertion/deletion", dG_insertion, dG_deletion)
    return dG_insertion, dG_deletion

def run_leg(args):
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

    inference: bool
        Whether or not we're in prediction mode

    Returns
    -------
    delta_G, delta_G grads
        delta G of the leg and its gradient

    """

    epoch_params, mol_name, mol, pdb, gpu_idx, inference = args
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
    return run_simulation(mol_name, mol, epoch_params, pdb, inference)

    # if not inference:
    #     dparams = np.zeros_like(epoch_params)
    #     dparams[combined_dp_idxs] = dG_grads
    # else:
    #     dparams = None

    # return dG, dparams


def train(protein_pdb, water_pdb, ligands, num_gpus):

    omm_forcefield = app.ForceField('amber99sb.xml', 'tip3p.xml') # for proteins

    # split train
    # np.random.shuffle(ligands)

    # train_frac = 0.75 # 75/25 split
    # train_frac = 1.0
    ds = Dataset(ligands)
    params = initialize_off_parameters()

    num_gpus = 1
    pool = multiprocessing.Pool(num_gpus)
    batch_size = num_gpus
    solvent_args = []
    complex_args = []
    for mol_idx, mol_data in enumerate(ds.iterbatches(batch_size)):
        mol, mol_name, true_dG = mol_data[0]
        solvent_args.append((params, mol_name+"_solvent_epoch", mol, water_pdb, mol_idx % batch_size, False))
        complex_args.append((params, mol_name+"_complex_epoch", mol, protein_pdb, mol_idx % batch_size, False))

    run_leg(complex_args[0])    
    # run_leg(solvent_args[0])
    assert 0
    # run_leg(solvent_args[0])


    complex_results = pool.map(run_leg, complex_args)
    for mol_idx, (dG_insertion, dG_deletion) in enumerate(complex_results):
        print("complex", mol_idx, "insertion", dG_insertion, "deletion", dG_deletion)

    solvent_results = pool.map(run_leg, solvent_args)
    for mol_idx, (dG_insertion, dG_deletion) in enumerate(solvent_results):
        print("solvent", mol_idx, "insertion", dG_insertion, "deletion", dG_deletion)

    assert 0

    train_dataset, test_dataset = ds.split(train_frac)

    print("Train/Test Split", len(train_dataset), len(test_dataset))
    # print("Loading parameters from previous run...")
    # starting_off_params = np.load('params_special_save.npy')
    # np.save('debug_starting_epoch', starting_off_params)

    lr = 5e-3
    opt_init, opt_update, get_params = optimizers.adam(lr)
    # opt_init, opt_update, get_params = optimizers.sgd(lr)

    opt_state = opt_init(starting_off_params)

    num_epochs = 100
    batch_size = num_gpus
    total_batches = 0

    pool = multiprocessing.Pool(num_gpus)

    itercount = itertools.count()
    for epoch in range(num_epochs):
    
        start_time = time.time()
        train_dataset.shuffle()

        epoch_loss = 0
        epoch_count = 0

        print("===============Epoch "+str(epoch)+"=============")

        # Train dataset
        for batch_idx, batch_data in enumerate(train_dataset.iterbatches(batch_size)):

            epoch_off_params = get_params(opt_state)
            np.save("train_params_epoch_"+str(epoch)+"_batch_"+str(batch_idx), epoch_off_params)

            complex_args = []
            solvent_args = []
            name_and_dGs = []

            for gpu_idx, (mol, mol_name, true_dG) in enumerate(batch_data):
                print("Processing batch", batch_idx, "mol", mol_name, "with true dG of", true_dG)
                complex_args.append((epoch_off_params, mol_name+"_complex_epoch_"+str(epoch), mol, protein_pdb, gpu_idx, False))
                solvent_args.append((epoch_off_params, mol_name+"_solvent_epoch_"+str(epoch), mol, water_pdb, gpu_idx, False))
                name_and_dGs.append((mol_name, true_dG))

            solvent_results = pool.map(run_leg, solvent_args)
            complex_results = pool.map(run_leg, complex_args)

            L2_grads = []

            for (mol_name, true_dG), (complex_dG, complex_dG_grads), (solvent_dG, solvent_dG_grads) in zip(name_and_dGs, complex_results, solvent_results):

                pred_dG = complex_dG - solvent_dG
                pred_dG_grad = complex_dG_grads - solvent_dG_grads

                L2_loss = (pred_dG-true_dG)**2
                L2_grad = 2*(pred_dG-true_dG)*pred_dG_grad

                print("dG grad max/min", np.amax(pred_dG_grad), np.amin(pred_dG_grad))
                L1_loss = np.abs(pred_dG-true_dG)
                # update on each epoch
                if np.any(np.isnan(L2_grad)) or np.any(np.isnan(L1_loss)):
                    print("ERROR: nans detected for", mol_name, "on epoch", epoch)
                else:
                    print(mol_name, "train l1 loss", L1_loss, "\t kJ/mol", "true vs pred", true_dG, "\t", pred_dG)
                    epoch_count += 1
                    epoch_loss += L1_loss
                    L2_grads.append(L2_grad)

            L2_grads = np.array(L2_grads)
            mean_L2_grads = np.mean(L2_grads, axis=0)

            opt_state = opt_update(next(itercount), mean_L2_grads, opt_state)
            epoch_off_params_after = get_params(opt_state)


        print("Avg Train Epoch L1 Loss: ", epoch_loss/epoch_count, "took", time.time()-start_time, "seconds")

        epoch_loss = 0
        epoch_count = 0

        epoch_off_params = get_params(opt_state)
        np.save("test_params_epoch_"+str(epoch), epoch_off_params)
        # Test Dataset
        for batch_idx, batch_data in enumerate(test_dataset.iterbatches(batch_size)):

            complex_args = []
            solvent_args = []
            name_and_dGs = []

            for gpu_idx, (mol, mol_name, true_dG) in enumerate(batch_data):
                print("Processing batch", batch_idx, "mol", mol_name, "with true dG of", true_dG)
                # True is for inference mode
                complex_args.append((epoch_off_params, mol_name+"_complex_epoch_"+str(epoch), mol, protein_pdb, gpu_idx, True))
                solvent_args.append((epoch_off_params, mol_name+"_solvent_epoch_"+str(epoch), mol, water_pdb, gpu_idx, True))
                name_and_dGs.append((mol_name, true_dG))

            complex_results = pool.map(run_leg, complex_args)
            solvent_results = pool.map(run_leg, solvent_args)

            for (mol_name, true_dG), (complex_dG, _), (solvent_dG, _) in zip(name_and_dGs, complex_results, solvent_results):
                pred_dG = complex_dG - solvent_dG
                L2_loss = (pred_dG-true_dG)**2
                L1_loss = np.abs(pred_dG-true_dG)

                if np.any(np.isnan(L1_loss)):
                    print("ERROR: nans detected for", mol_name, "on epoch", epoch)
                else:
                    print(mol_name, "test l1 loss", L1_loss, "\t kJ/mol", "true vs pred", true_dG, "\t", pred_dG)
                    epoch_count += 1
                    epoch_loss += L1_loss

        print("Avg Test Epoch L1 Loss: ", epoch_loss/epoch_count)

    pool.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Differentiable Absolute Binding Free Energy')
    parser.add_argument('--num_gpus', type=int, help='number of gpus to train on')
    parser.add_argument('--ligands_sdf', type=str, help='sdf file pointing to ligands. Please set IC50[uM] and Name properties.')
    parser.add_argument('--protein_pdb', type=str, help='pdb file pointing to belly-solvated apo protein structure.')
    parser.add_argument('--water_pdb', type=str, help='pdb file pointing to ligand centered water shell.')

    args = parser.parse_args()

    # both of these need to be centered around the ligand
    protein_pdb = app.PDBFile(args.protein_pdb)
    protein_pdb.filepath = args.protein_pdb
    water_pdb = app.PDBFile(args.water_pdb)
    water_pdb.filepath = args.water_pdb

    # build dataset
    suppl = Chem.SDMolSupplier(args.ligands_sdf, removeHs=False)
    ligands = []
    for idx, mol in enumerate(suppl):
        if idx > 0:
            break
        name = mol.GetProp('Name')
        uIC50 = float(mol.GetProp('IC50[uM]'))
        binding_kJ_per_mole = convert_uIC50_to_kJ_per_mole(uIC50)
        unbinding_kJ_per_mole = binding_kJ_per_mole*-1
        ligands.append((mol, name, unbinding_kJ_per_mole))

    print("Arguments passed", args)

    train(protein_pdb, water_pdb, ligands, args.num_gpus)
