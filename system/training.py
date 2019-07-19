import os
import sys
import numpy as np
import time
import datetime
import sklearn.metrics
import jax
import scipy
import json
import glob
import csv
import itertools
import functools

from scipy import stats
from rdkit import Chem
from rdkit.Chem import AllChem

from system import serialize
from system import forcefield
from system import simulation

from openforcefield.typing.engines.smirnoff import ForceField
from timemachine.observables import rmsd
from timemachine.lib import custom_ops
from jax.experimental import optimizers, stax

import multiprocessing

def run_test(num_epochs,
             testing_params):
    training_data = np.load(properties['training_data'],allow_pickle=True)['data']
    training_data = training_data[400:]
        
    batch_size = properties['batch_size']
    pool = multiprocessing.Pool(batch_size)
    num_data_points = len(training_data)
    num_batches = int(np.ceil(num_data_points/batch_size))
        
    for epoch in range(num_epochs):
        
        start_time = time.time()

        print('--- testing',epoch, "started at", datetime.datetime.now(), '----')

        losses = []

        for b_idx in range(num_batches):            
            start_idx = b_idx*batch_size
            end_idx = min((b_idx+1)*batch_size, num_data_points)
            batch_data = training_data[start_idx:end_idx]

            args = []

            for b_idx, b in enumerate(batch_data):
                args.append([testing_params,b[0],b[1],b_idx])

            results = pool.map(rmsd_run,args)

            for _, loss in results:
                if not np.isnan(loss):
                    losses.append(loss)

        losses = np.array(losses)
        mean_loss = np.mean(losses)

        np.savez('test_{}.npz'.format(epoch), loss=losses, params=testing_params)

        print('''
    Test: {}
    ==============
    Mean RMSD: {}
    Elapsed time: {} seconds
    ==============
            '''.format(epoch,mean_loss,time.time()-start_time))
        
    return losses

def average_derivs(R, label):
    running_sum_derivs = None
    running_sum_confs = None

    n_reservoir = len(R)
    
    if properties['run_type'] == 'train':
        grad_fun = jax.jit(jax.grad(rmsd.opt_rot_rmsd,argnums=0))
    
    for E, dE_dx, dx_dp, dE_dp, x in R:
        if running_sum_derivs is None:
            running_sum_derivs = np.zeros_like(dE_dp)
        if running_sum_confs is None:
            running_sum_confs = np.zeros_like(x)
           
        if properties['run_type'] == 'train':
            grad_conf = grad_fun(x,label)
            combined_grad = np.einsum('kl,mkl->m', grad_conf, dx_dp)
            running_sum_derivs += combined_grad
            
        running_sum_confs += x
        
    if properties['run_type'] == 'train':
        print(np.amax(abs(running_sum_derivs/n_reservoir)) * properties['learning_rate'])
        if np.isnan(running_sum_derivs/n_reservoir).any() or np.amax(abs(running_sum_derivs/n_reservoir)) * properties['learning_rate'] > 1e-2 or np.amax(abs(running_sum_derivs/n_reservoir)) == 0:
            print("bad gradients/nan energy")
            running_sum_derivs = np.zeros_like(running_sum_derivs)
            running_sum_confs *= np.nan

    return running_sum_derivs/n_reservoir, running_sum_confs/n_reservoir

def rmsd_run(params):
        
    p = multiprocessing.current_process()
    smirnoff_params, guest_sdf_file, label, idx = params
    os.environ['CUDA_VISIBLE_DEVICES'] = str(idx % properties['batch_size'])
    
    derivs = np.zeros_like(smirnoff_params)
    losses = []

    print('processing',guest_sdf_file)
    
    guest_sdf = open(os.path.join(properties['guest_directory'], guest_sdf_file), "r").read()
    smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")
    
    mol = Chem.MolFromMol2Block(guest_sdf, sanitize=True, removeHs=False, cleanupSubstructures=True)
    
    # embed some bad conformers
    if properties['random'] == 'yes':
        AllChem.EmbedMultipleConfs(mol,numConfs=1, clearConfs=True, useExpTorsionAnglePrefs=False, useBasicKnowledge=False)
    else:
        AllChem.EmbedMultipleConfs(mol,numConfs=1, randomSeed=1234,clearConfs=True, useExpTorsionAnglePrefs=False, useBasicKnowledge=False)
    
    smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")

    guest_potentials, _, smirnoff_param_groups, _, guest_masses = forcefield.parameterize(mol, smirnoff)

    for conf_idx in range(mol.GetNumConformers()):
        c = mol.GetConformer(conf_idx)
        conf = np.array(c.GetPositions(),dtype=np.float64)
        guest_conf = conf/10

        def filter_groups(param_groups, groups):
            roll = np.zeros_like(param_groups)
            for g in groups:
                roll = np.logical_or(roll, param_groups == g)
            return roll

        dp_idxs = properties['dp_idxs']
        
        if len(dp_idxs) == 0:
            guest_dp_idxs = np.array([0])
        else:
            guest_dp_idxs = np.argwhere(filter_groups(smirnoff_param_groups, dp_idxs)).reshape(-1)
        
        RG = simulation.run_simulation(
            guest_potentials,
            smirnoff_params,
            smirnoff_param_groups,
            guest_conf,
            guest_masses,
            guest_dp_idxs,
            100
        )
        
        G_deriv, G_conf = average_derivs(RG, label)

        loss = rmsd.opt_rot_rmsd(G_conf,label)

        derivs[guest_dp_idxs] += G_deriv
        losses.append(loss)

    losses = np.array(losses)

    return derivs, np.mean(losses)

def train_rmsd(num_epochs,
               opt_init,
               opt_update,
               get_params,
               init_params):
    
#     training_data = []
    
#     smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")
    
#     csd_dir = '/home/ubuntu/Relay/structures/CSD'
# #     count = 0
    
#     bad_mols = [
#     'ACEONP.mol2',
#     'ABINOR03.mol2',
#     'ABEPUS.mol2',
#     'ABIKIF.mol2',
#     'ACEPAB.mol2',
#     'ACEKIC.mol2',
#     'ACEDAC01.mol2',
#     'ACESTC.mol2',
#     'ACIGUP.mol2',
#     'ACESTA.mol2',
#     'ACEKUO.mol2',
#     'ABIMAZ.mol2',
#     'ACIGRA.mol2',
#     'ACAPUP.mol2',
#     'ABINOR02.mol2',
#     'ACIQUB.mol2',
#     'AAPYPE.mol2',
#     'ABRTOL.mol2',
#     'ACESTB.mol2',
#     'ABMHFO.mol2',
#     'ABECEP.mol2',
#     'ABILAC10.mol2',
#     'ABVPRO.mol2',
#     'ACAVAB.mol2',
#     'ABATRG.mol2',
#     'AARBOX.mol2',
#     'ACDANT.mol2',
#     'ABEPEC.mol2',
#     'ABGPON.mol2',
#     'ABSCIC.mol2',
#     'ABEDOA.mol2',
#     'ACBTHO.mol2',
#     'ACBRCN.mol2',
#     'ABAZOS01.mol2',
#     'ACAPAL.mol2',
#     'ACETAC08.mol2',
#     'ABORUE01.mol2',
#     'ACIKON.mol2',
#     'ABTOET.mol2',
#     'ABZIOX.mol2',
#     'ACARDL.mol2',
#     'ACAZOT.mol2',
#     'ABEQAZ.mol2',
#     'ACETTP.mol2',
#     'ABPACH10.mol2',
#     'ABRAHE.mol2',
#     'ACEOXM.mol2',
#     'ACIJOM.mol2',
#     'ACETAC.mol2',
#     'ACCMTS.mol2',
#     'ACASAY.mol2',
#     'ACGRAY.mol2',
#     'ACEDAC10.mol2',
#     'ACANAC10.mol2',
#     'ACHPQZ.mol2',
#     'ACIHEA.mol2',
#     'ABOGUW.mol2',
#     'ACHOLB.mol2',
#     'ABMIAL.mol2',
#     'ABSBPP.mol2',
#     'ABAZUY.mol2',
#     'ABCMHP.mol2',
#     'ABUBAA01.mol2',
#     'ACDOME.mol2',
#     'ABEBAK.mol2',
#     'ABMPAZ.mol2'
#     ]
    
#     for filename in os.listdir(csd_dir):
# #     filename = 'ABAZOS01.mol2'
#         try:
#             if filename in bad_mols:
#                 raise Exception
#             print(filename)
#             structure_file = open(csd_dir + '/' + filename,'r').read()
# #             smiles = structure_file.partition('\n')[0][1:]
#             ref_mol = Chem.MolFromMol2Block(structure_file, sanitize=True, removeHs=False, cleanupSubstructures=True)
#             c = ref_mol.GetConformer(0)
#             conf = np.array(c.GetPositions(), dtype=np.float64)
#             guest_conf = conf/10 # convert to md_units
#             training_data.append([filename,guest_conf])
#         except:
#             print('bad mol2')

# #     smiles_array = [
# #         'CCCC[NH3+]',
# #         'CCCCCC[NH3+]',
# #         'CCCCCCCC[NH3+]',
# #         'OC1CCCC1',
# #         'OC1CCCCCC1',
# #         'CCCC([O-])=O',
# #         'CCCCCC([O-])=O',
# #         'CCCCCCCC([O-])=O',
# #         'CCCC[NH2+]C',
# #         'CCCC(C)[NH3+]',
# #         'CCCCC[NH3+]',
# #         'CCCCCC[NH2+]C',
# #         'CCCCCC(C)[NH3+]',
# #         'CCCCCCC[NH3+]',
# #         'CCCCCCC(C)[NH3+]',
# #         'OC1CCC1',
# #         'OC1CCCCCCC1',
# #         'CCCCC([O-])=O',
# #         'CCC/C=C/C([O-])=O',
# #         'CC/C=C/CC([O-])=O',
# #         'CCCCCCC([O-])=O',
# #         '[O-]C(=O)CCCCC=C'
# #     ]

#     smiles_array = np.load('freesolv_smiles.npz',allow_pickle=True)['smiles']
    
#     bad_mols = ['C(F)(F)Cl', 'N', 'CCl', 'C(Cl)(Cl)(Cl)Cl', 'CBr', 'S', 'C(Cl)(Cl)Cl', 'C=O', 'C', 'C(F)(F)(F)Br', 'C(Cl)Cl', 'C(Br)Br', 'CF', 'C(F)(F)(F)F', 'C(F)Cl', 'C(Br)(Br)Br', 'C(I)I', 'CI']
    
#     for smiles in smiles_array:
        
#         if smiles in bad_mols:
#             continue

#         ref_mol = Chem.MolFromSmiles(smiles)
#         ref_mol = Chem.AddHs(ref_mol)
#         AllChem.EmbedMolecule(ref_mol)
        
#         c = ref_mol.GetConformer(0)
#         conf = np.array(c.GetPositions(),dtype=np.float64)
#         guest_conf = conf/10

# #         smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")

# #         guest_potentials, smirnoff_params, smirnoff_param_groups, guest_conf, guest_masses = forcefield.parameterize(ref_mol, smirnoff)

# #         RG = simulation.run_simulation(
# #                 guest_potentials,
# #                 smirnoff_params,
# #                 smirnoff_param_groups,
# #                 guest_conf,
# #                 guest_masses,
# #                 np.array([0]),
# #                 1000
# #             )

#         training_data.append([smiles,guest_conf])

#     np.savez('training_data.npz',data=training_data)
    
# #     assert 0
    
    # training data for RMSD must be .npz file
    training_data = np.load(properties['training_data'],allow_pickle=True)['data']
    training_data = training_data[:128]
    
#     bad_smiles = [
# 'BrC1C2CC3CC1CB(C3)(C2)O1CCCC1',
# 'CCCC(=O)OC1C(O)C(OC(C)=O)C2(C)C(CC(O)C(C)C2C(OC(C)=O)C23OC2(C)C(=O)OC3C=C1C)OC(C)=O',
# 'CCC(CC)(C(N(Cc1ccccc1)C1OC(COC(=O)C(C)(C)C)C(OC(=O)C(C)(C)C)C(OC(=O)C(C)(C)C)C1OC(=O)C(C)(C)C)c1ccccc1)C(=O)OC',
# 'CC(=O)NC1=NC23CCCN2C(=O)C2=CC(=C(Br)N2C3N1)Br',
# 'Fc1ccc2OB(c3ccccc3)(c3ccccc3)N3=C(SC4=C3SC3=N4B(Oc4ccc(F)cc34)(c3ccccc3)c3ccccc3)c2c1',
# 'CCCCCCCCOc1cc(C=Cc2ccc(cc2)C=Cc2ccccc2C#N)c(OCCCCCCCC)cc1C=Cc1ccc(cc1)C=Cc1ccccc1C#N',
# 'CC1(CC(O)=O)c2ccccc2OC1(c1ccc(F)cc1)c1ccc(F)cc1',
# 'CC(=O)NCCNC(C)=O',
# 'CC(=O)NCCCCCNC(C)=O',
# 'O1c2ccccc2C2=N(C3=C(S2)N2=C(S3)c3ccccc3OB2(c2ccccc2)c2ccccc2)B1(c1ccccc1)c1ccccc1',
# 'CC(=O)NCCCCNC(C)=O',
# 'CC(=O)OCC1OC(OC2C(COC(c3ccccc3)(c3ccccc3)c3ccccc3)OC(OC(C)=O)C(OC(C)=O)C2OC(C)=O)C(OC(C)=O)C(OC(C)=O)C1OC(C)=O',
# 'c1ccc(cc1)N1C=CN(=C1)B(c1ccccc1)(c1ccccc1)c1ccccc1',
# 'CC(C)CCCC(C)C1CCC2C3CCC4CC(Cl)C(Cl)CC4(C)C3CCC12C',
# 'COC1C2CC(=O)OC(C)CC=CC=CC(OC(C)=O)C(C)CC(C1OC(C)=O)C2C=O',
# 'COP(=O)(CC(O)CN(C(C)c1ccccc1)C(C)c1ccccc1)OC',
# 'CCC1(C)C=C2CCC3C(C)(C)C(CCC3(C)C2C(Br)C1=O)OC(C)=O',
# 'CC(=O)NCC1(CCN(CCCC(=O)c2ccc(F)cc2)CC1)c1ccccc1',
# 'CC(=O)OC1CCC2C3CCC4CC(=O)C(Br)CC4C3(C)CCC12C',
# 'CCc1c(CC)c(CC)c2c(c1CC)c(c1ccccc1)c(c1ccccc1)c(c1ccccc1)c2c1ccccc1',
# 'COc1ccc2OB(c3ccccc3)(c3ccccc3)N3=C(SC4=C3SC3=N4B(Oc4ccc(OC)cc34)(c3ccccc3)c3ccccc3)c2c1',
# 'CC(=O)OCC1OC=C(C(OC(C)=O)C1OC(C)=O)N(C(C)=O)C(C)=O',
# 'CC1(C)CN(C(=O)C1OC(=O)C12CCC(C)(C(=O)O1)C2(C)C)c1ccc(cc1)C(=O)OCc1ccccc1',
# 'CC(=O)OC1C(Cl)C(O)(CCl)C(OC(C)=O)C(OC(C)=O)C1OC(C)=O',
# 'FC1CCCCC1OC(=O)C(=Cc1ccccc1)c1ccccc1',
# 'CCCCCCCCOc1cc(C=Cc2ccc(cc2)C=C(C#N)c2ccc(OC)cc2)c(OCCCCCCCC)cc1C=Cc1ccc(cc1)C=C(C#N)c1ccc(OC)cc1',
# 'CC(=O)OCC1OC(OC(C)=O)C(OC(C)=O)C(OC(C)=O)C1OC1OC(COC(C)=O)C(OC(C)=O)C(OC(C)=O)C1OC(C)=O',
# 'Cc1cc2c3ccccc3OB(c3ccccc3)(c3ccccc3)n2c2ccccc12',
# 'FB12Oc3ccccc3c3cccc(c4ccccc4O1)n23',
# 'CC(=O)Nc1cccc2c1c1ccccc1c(c1ccc(cc1)C(F)(F)F)c2c1ccc(cc1)C(F)(F)F',
# 'CC(=O)C1=C(C=CS1)C(O)=O',
# 'Fc1c(F)c(F)c(c(F)c1F)C1=C(C(=CC1)c1c(F)c(F)c(F)c(F)c1F)c1c(F)c(F)c(F)c(F)c1F',
# 'O=C1N(N2C(=O)c3ccccc3N=C2CSCC#C)C(=Nc2ccccc12)CSCC#C',
# 'c1ccc(cc1)B(c1ccccc1)(c1ccccc1)n1ccncc1',
# 'C1SCC2=C1SC(S2)=C1SC2=CSC=C2S1',
# 'CC(C)CCCC(C)C1CCC2C3CCC4CC(Br)C(Br)CC4(C)C3CCC12C',
# 'CC(=O)OC12CC=CCC1(Cl)CC=CC2',
# 'CCOC(=O)C1(CCCCC1)C(N(CC)C1OC(COC(=O)C(C)(C)C)C(OC(=O)C(C)(C)C)C(OC(=O)C(C)(C)C)C1OC(=O)C(C)(C)C)c1ccc(cc1)N(=O)=O',
# 'COC(=O)C(NC(=O)C(NC(=O)C(C)NC(=O)OC(C)(C)C)=Cc1cccc2ccccc12)C(C)C',
# 'CCCOc1c2Cc3cc(cc(Cc4cc(cc(C(C(=O)N(C)C)c5cc(cc(Cc1cc(c2)C(C)(C)C)c5O)C(C)(C)C)c4OCC(=O)NC(C)c1ccccc1)C(C)(C)C)c3O)C(C)(C)C',
# 'CC(=O)NCCCNC(C)=O',
# 'CC(C)(O)C=C1OC(=O)C(=C1)C1CCC23CC12CCC1C2(C)C=CC(=O)C(C)(C)C2CC(O)C31C',
# 'CC(C)C(=O)OC1C(O)C2C(OC(=O)C2=C)C=C(C)CCC=C1C=O',
# 'CCOC1CC(=O)C2(C)C3CCC45C(CCC4C(C)(OC5OCC)C4CC(=C(C)C(=O)O4)C)C3CC3OC23C1O'
#     ]
    
#     print(len(training_data))
#     delete = []
#     for i in range(len(training_data)):
#         structure_file = open(csd_dir + '/' + training_data[i,0],'r').read()
#         smiles = structure_file.partition('\n')[0][1:]
#         if smiles in bad_smiles:
#             delete.append(i)
#     training_data = np.delete(training_data, delete, 0)
#     print(len(training_data))
#     print(len(bad_smiles))
            
#     np.savez('training_data.npz',data=training_data)

#     assert 0
            
#     for smiles, conf in training_data:
#         try:
#             mol = Chem.MolFromSmiles(smiles)
#             mol = Chem.AddHs(mol)
#             AllChem.EmbedMultipleConfs(mol,numConfs=1, randomSeed=1234,clearConfs=True, useExpTorsionAnglePrefs=False, useBasicKnowledge=False)
#             c = mol.GetConformer(0)
#             c_2 = np.array(c.GetPositions(),dtype=np.float64)
#             guest_conf = c_2/10
#             loss = rmsd.opt_rot_rmsd(guest_conf,conf)
#         except:
#             print(smiles)
#     assert 0
        
    batch_size = properties['batch_size']
    pool = multiprocessing.Pool(batch_size)
    num_data_points = len(training_data)
    num_batches = int(np.ceil(num_data_points/batch_size))
    
    opt_state = opt_init(init_params)
    count = 0
        
    for epoch in range(num_epochs):
        
        start_time = time.time()

        print('--- epoch:', epoch, "started at", datetime.datetime.now(), '----')
        np.random.shuffle(training_data)
        
        losses = []
        
        for b_idx in range(num_batches):            
            start_idx = b_idx*batch_size
            end_idx = min((b_idx+1)*batch_size, num_data_points)
            batch_data = training_data[start_idx:end_idx]

            args = []
    
            for b_idx, b in enumerate(batch_data):
                args.append([get_params(opt_state),b[0],b[1],b_idx])
                
            results = pool.map(rmsd_run,args)
            
            batch_dp = np.zeros_like(get_params(opt_state))
            
            for grad, loss in results:
                batch_dp += grad
                if not np.isnan(loss):
                    losses.append(loss)

            opt_state = opt_update(count, batch_dp, opt_state)
            count += 1
        
        losses = np.array(losses)
        mean_loss = np.mean(losses)
        
        np.savez('run_{}.npz'.format(epoch), loss=losses,params=get_params(opt_state))
        
        print('''
Epoch: {}
==============
Mean RMSD: {}
Elapsed time: {} seconds
==============
            '''.format(epoch,mean_loss,time.time()-start_time))
        
    return losses, get_params(opt_state)
        
def rescale_and_center(conf, scale_factor=1):
    mol_com = np.sum(conf, axis=0)/conf.shape[0]
    true_com = np.array([1.97698696, 1.90113478, 2.26042174]) # a-cd
#     true_com = np.array([5.4108882, 4.75821426, 9.33421262]) # london
    centered = conf - mol_com  # centered to origin
    return true_com + centered/scale_factor
    
def run_simulation(params):

#     p = multiprocessing.current_process()
    combined_params, guest_sdf_file, label, idx = params
    label = float(label)

    if 'gpu_offset' in properties:
        gpu_offset = properties['gpu_offset']
    else:
        gpu_offset = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(idx % properties['batch_size'] + gpu_offset)

    host_potentials, host_conf, (dummy_host_params, host_param_groups), host_masses = serialize.deserialize_system(properties['host_path'])
    host_params = combined_params[:len(dummy_host_params)]
    guest_sdf = open(os.path.join(properties['guest_directory'], guest_sdf_file), "r").read()
    print("processing",guest_sdf_file)
        
    smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")
    
    mol = Chem.MolFromMol2Block(guest_sdf, sanitize=True, removeHs=False, cleanupSubstructures=True)
    AllChem.EmbedMultipleConfs(mol, numConfs=10, randomSeed=1234, clearConfs=True)
#     AllChem.EmbedMultipleConfs(mol, numConfs=1, clearConfs=True)

    guest_potentials, _, smirnoff_param_groups, guest_conf, guest_masses = forcefield.parameterize(mol, smirnoff)
    smirnoff_params = combined_params[len(dummy_host_params):]

    RH = []
    RG = []
    RHG = []
    
    for conf_idx in range(mol.GetNumConformers()):
        c = mol.GetConformer(conf_idx)
        conf = np.array(c.GetPositions(),dtype=np.float64)
        guest_conf = conf/10
#         rot_matrix = stats.special_ortho_group.rvs(3).astype(dtype=np.float32)
#         guest_conf = np.matmul(guest_conf, rot_matrix)
        guest_conf = rescale_and_center(guest_conf, scale_factor=4)
        combined_potentials, _, combined_param_groups, combined_conf, combined_masses = forcefield.combiner(
            host_potentials, guest_potentials,
            host_params, smirnoff_params,
            host_param_groups, smirnoff_param_groups,
            host_conf, guest_conf,
            host_masses, guest_masses)

        num_atoms = len(combined_masses)

        def filter_groups(param_groups, groups):
            roll = np.zeros_like(param_groups)
            for g in groups:
                roll = np.logical_or(roll, param_groups == g)
            return roll

        dp_idxs = properties['dp_idxs']

        if len(dp_idxs) == 0:
            host_dp_idxs = np.array([0])
            guest_dp_idxs = np.array([0])
            combined_dp_idxs = np.array([0])
        else:
            host_dp_idxs = np.argwhere(filter_groups(host_param_groups, dp_idxs)).reshape(-1)
            guest_dp_idxs = np.argwhere(filter_groups(smirnoff_param_groups, dp_idxs)).reshape(-1)
            combined_dp_idxs = np.argwhere(filter_groups(combined_param_groups, dp_idxs)).reshape(-1)

        RH_i = None
        if properties['fit_method'] == 'absolute':
            RH_i = simulation.run_simulation(
                host_potentials,
                host_params,
                host_param_groups,
                host_conf,
                host_masses,
                host_dp_idxs,
                1000
            )

        RG_i = simulation.run_simulation(
            guest_potentials,
            smirnoff_params,
            smirnoff_param_groups,
            guest_conf,
            guest_masses,
            guest_dp_idxs,
            1000
        )

        # Don't add anything to reservoir if energy is nan
        if np.isnan(RG_i[0][0]):
            continue
        
        RHG_i = simulation.run_simulation(
            combined_potentials,
            combined_params,
            combined_param_groups,
            combined_conf,
            combined_masses,
            combined_dp_idxs,
            1000
        )
        
        # Don't add anything to reservoir if energy is nan
        if np.isnan(RHG_i[0][0]):
            continue
            
        if RH_i is not None:
            RH.append(RH_i[0])
            
        RG.append(RG_i[0])
        RHG.append(RHG_i[0])
    
    # RH, RG, and RHG are reservoirs containing N conformers each
    return RH, RG, RHG, label, host_dp_idxs, guest_dp_idxs, combined_dp_idxs, num_atoms

def boltzmann_derivatives(reservoir):
    
    # if reservoir is empty, return
    if len(reservoir) == 0:
        return 0,0
    
    n_reservoir = len(reservoir)
    num_atoms = len(reservoir[0][-1])

    E= []
    dE_dx_temp = np.zeros((n_reservoir,n_reservoir,num_atoms,3))
    dE_dx = []
    dx_dp = []
    dE_dp = []
    
    # energies and derivatives for each conformer
    for E_i, dE_dx_i, dx_dp_i, dE_dp_i, _ in reservoir:
        E.append(E_i)
        dE_dx.append(dE_dx_i)
        dx_dp.append(dx_dp_i)
        dE_dp.append(dE_dp_i)
    
    E = np.array(E,dtype=np.float32)
    dE_dx = np.array(dE_dx,dtype=np.float64)
    dx_dp = np.array(dx_dp)
    dx_dp = np.transpose(dx_dp,(0,2,3,1))
    dE_dp = np.array(dE_dp) 
    
    for i in range(n_reservoir):
        dE_dx_temp[i][i] = np.array(dE_dx[i])
        
    dE_dx = dE_dx_temp
    
    ds_de_fn = jax.jacfwd(stax.softmax, argnums=(0,))
    ds_de = ds_de_fn(-E)

    # tensor contract [C,C,N,3] with [C,N,3,P] and dE_dp
    tot_dE_dp = np.einsum('ijkl,jklm->im', dE_dx, dx_dp) + dE_dp
    s_e = stax.softmax(-E)

    total_derivs = np.matmul(-ds_de[0], tot_dE_dp)*np.expand_dims(E, 1) + np.expand_dims(s_e, axis=-1) * tot_dE_dp

    # Might need to use -E/kT as input to softmax instead?
    return np.sum(stax.softmax(-E)*E), np.sum(total_derivs, axis=0)

def compute_derivatives(params1,
                        params2,
                        host_params,
                       combined_params):
    
    RH1, RG1, RHG1, label_1, host_dp_idxs, guest_dp_idxs_1, combined_dp_idxs_1, num_atoms = params1
    G_E, G_derivs = boltzmann_derivatives(RG1)
    HG_E, HG_derivs = boltzmann_derivatives(RHG1)
#     G_E, G_derivs, _ = simulation.average_E_and_derivatives(RG1)
#     HG_E, HG_derivs, _ = simulation.average_E_and_derivatives(RHG1)
        
    if properties['fit_method'] == 'absolute':
#         H_E, H_derivs, _ = simulation.average_E_and_derivatives(RH1)
        H_E, H_derivs = boltzmann_derivatives(RH1)
        pred_enthalpy = HG_E - (G_E + H_E)
        label = label_1
        delta = pred_enthalpy - label

        combined_derivs = np.zeros_like(combined_params)
        combined_derivs[combined_dp_idxs_1] += HG_derivs
        combined_derivs[host_dp_idxs] -= H_derivs
        combined_derivs[guest_dp_idxs_1 + len(host_params)] -= G_derivs
            
    elif properties['fit_method'] == 'relative':
        RH2, RG2, RHG2, label_2, host_dp_idxs, guest_dp_idxs_2, combined_dp_idxs_2, _ = params2
        G_E_2, G_derivs_2 = boltzmann_derivatives(RG2)
        HG_E_2, HG_derivs_2 = boltzmann_derivatives(RHG2)
#         G_E_2, G_derivs_2, _ = simulation.average_E_and_derivatives(RG2)
#         HG_E_2, HG_derivs_2, _ = simulation.average_E_and_derivatives(RHG2)
        
        pred_enthalpy = HG_E - HG_E_2 - G_E + G_E_2
        label = label_1 - label_2
        delta = pred_enthalpy - label
        
        combined_derivs = np.zeros_like(combined_params)
        
        # if reservoir is empty, derivatives are all zero
        if type(G_derivs_2) is np.ndarray:
            combined_derivs[combined_dp_idxs_1] += HG_derivs
            combined_derivs[combined_dp_idxs_2] -= HG_derivs_2
            combined_derivs[guest_dp_idxs_1 + len(host_params)] -= G_derivs
            combined_derivs[guest_dp_idxs_2 + len(host_params)] += G_derivs_2
        
    if properties['loss_fn'] == 'L2':
        '''
        loss = (delta) ^ 2
        '''
        combined_derivs = 2*delta*combined_derivs
    elif properties['loss_fn'] == 'L1':
        '''
        loss = |delta|
        '''
        combined_derivs = (delta/np.abs(delta))*combined_derivs
    elif properties['loss_fn'] == 'Huber':
        '''
        if |delta| < cutoff, loss = (delta^2) / (2 * cutoff)
        if |delta| > cutoff, loss = |delta|
        '''
        huber_cutoff = np.where('huber_cutoff' in properties, properties['huber_cutoff'], 1)
        combined_derivs = np.where(abs(delta) < huber_cutoff, delta / huber_cutoff, delta/np.abs(delta)) * combined_derivs
    elif properties['loss_fn'] == 'log-cosh':
        '''
        loss = log(cosh(delta))
        '''
        combined_derivs = (1 / np.cosh(delta)) * np.sinh(delta) * combined_derivs
        
    print(np.amax(abs(combined_derivs)) * properties['learning_rate'])
    if np.isnan(combined_derivs).any() or np.amax(abs(combined_derivs)) * properties['learning_rate'] > 1e-2 or np.amax(abs(combined_derivs)) == 0:
        print("bad gradients/nan energy")
        combined_derivs = np.zeros_like(combined_derivs)
        pred_enthalpy = np.nan

    return combined_derivs, pred_enthalpy, label
    
def initialize_parameters(host_path):

    # setting general smirnoff parameters for guest
    # random smiles string to initialize parameters
    ref_mol = Chem.MolFromSmiles('CCCC')
    ref_mol = Chem.AddHs(ref_mol)
    AllChem.EmbedMolecule(ref_mol)

    smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")

    _, smirnoff_params, _, _, _ = forcefield.parameterize(ref_mol, smirnoff)
                  
#     structure_path = os.path.join(properties['guest_directory'], properties['guest_template'])
#     if '.mol2' in properties['guest_template']:
#         structure_file = open(structure_path,'r').read()
#         ref_mol = Chem.MolFromMol2Block(structure_file, sanitize=True, removeHs=False, cleanupSubstructures=True)
#     else:
#         raise Exception('only mol2 files currently supported for ligand training')

    if properties['loss_type'] == 'Enthalpy':
        _, _, (host_params, _), _ = serialize.deserialize_system(host_path)       
        epoch_combined_params = np.concatenate([host_params, smirnoff_params])
    else:
        epoch_combined_params = None
    
    return epoch_combined_params, smirnoff_params

def train(num_epochs, 
          opt_init, 
          opt_update, 
          get_params, 
          init_params):
    
    data_file = open(properties['training_data'],'r')
    data_reader = csv.reader(data_file, delimiter=',')
    training_data = list(data_reader)
           
    batch_size = properties['batch_size']
    pool = multiprocessing.Pool(batch_size)
    if properties['fit_method'] == 'relative':
        batch_size -= 1
    num_data_points = len(training_data)
    num_batches = int(np.ceil(num_data_points/batch_size))

    opt_state = opt_init(init_params)
    count = 0
    
    _, _, (dummy_host_params, host_param_groups), _ = serialize.deserialize_system(properties['host_path'])
    
    for epoch in range(num_epochs):

        print('--- epoch:', epoch, "started at", datetime.datetime.now(), '----')
        np.random.shuffle(training_data)
        
        epoch_predictions = []
        epoch_labels = []
        epoch_filenames = []

        for fn in training_data:
            epoch_filenames.append(fn[0])
        
        for b_idx in range(num_batches):
            start_idx = b_idx*batch_size
            end_idx = min((b_idx+1)*batch_size, num_data_points)
            batch_data = training_data[start_idx:end_idx]

            args = []

            if properties['fit_method'] == 'relative':
                args.append([get_params(opt_state), properties['relative_reference'][0], properties['relative_reference'][1], 0])
            
            for b_idx, b in enumerate(batch_data):
                if properties['fit_method'] == 'relative':
                    args.append([get_params(opt_state), b[0], b[1], b_idx + 1])
                else:
                    args.append([get_params(opt_state), b[0], b[1], b_idx])
            
            results = pool.map(run_simulation, args)
            
            final_results = []
            if properties['fit_method'] == 'absolute':
                for params in results:
                    final_results.append(compute_derivatives(params, None, dummy_host_params, get_params(opt_state)))
            if properties['fit_method'] == 'relative':
                params1 = results[0]
                for i in range(1,len(results)):
                    final_results.append(compute_derivatives(params1, results[i], dummy_host_params, get_params(opt_state)))
                    
            batch_dp = np.zeros_like(init_params)

            for grads, preds, labels in final_results:
                batch_dp += grads
                if not np.isnan(preds):
                    epoch_predictions.append(preds)
                    epoch_labels.append(labels)
                    
            # if everything is nan, terminate simulation
            if len(epoch_predictions) == 0:
                print("all energies are nan")
                return
            
            count += 1
            opt_state = opt_update(count, batch_dp, opt_state)

        epoch_predictions = np.array(epoch_predictions)
        epoch_labels = np.array(epoch_labels)

        np.savez("run_{}.npz".format(epoch), preds=epoch_predictions, labels=epoch_labels, filenames=epoch_filenames, params=get_params(opt_state))
    
        mae = np.mean(np.abs(epoch_predictions-epoch_labels))
        mean = np.mean(epoch_predictions-epoch_labels)
        if len(epoch_predictions) > 1:
            pearson_r = stats.pearsonr(epoch_predictions, epoch_labels)
            r2_score = sklearn.metrics.r2_score(epoch_predictions, epoch_labels)
            print('''
Epoch: {}
=================
Pearson R: {}
R2 score: {}
MAE: {}
Mean: {}
=================
            '''.format(epoch,pearson_r[0], r2_score, mae, mean))
        else:
            print(''' 
Epoch: {}
=================
MAE: {}
Mean: {}
=================
            '''.format(epoch,mae, mean))
        
    return preds, labels, get_params(opt_state)

def initialize_optimizer(optimizer, 
                         lr):
    
    if optimizer == 'Adam':
        opt_init, opt_update, get_params = optimizers.adam(lr)
    elif optimizer == 'SGD':
        opt_init, opt_update, get_params = optimizers.sgd(lr)
    elif optimizer == 'RMSProp':
        opt_init, opt_update, get_params = optimizers.rmsprop(lr)
    elif optimizer == 'Adagrad':
        opt_init, opt_update, get_params = optimizers.adagrad(lr)
    elif optimizer == 'SM3':
        opt_init, opt_update, get_params = optimizers.sm3(lr)
        
    return opt_init, opt_update, get_params

if __name__ == "__main__":
    
    config_file = glob.glob('*.json')
    if len(config_file) == 0:
        raise Exception('config file not found')
    elif len(config_file) > 1:
        raise Exception('multiple config files found')
    with open(config_file[0], 'r') as file:
        config = json.load(file)
   
    properties = config
    
    if properties['run_type'] == 'train':
        opt_init, opt_update, get_params = initialize_optimizer(properties['optimizer'], properties['learning_rate'])

        if properties['loss_type'] == 'RMSD':
            _, init_params = initialize_parameters(None)
            losses, final_params = train_rmsd(properties['num_epochs'],opt_init,opt_update,get_params,init_params)

        elif properties['loss_type'] == 'Enthalpy':
            init_params, _ = initialize_parameters(properties['host_path'])
            np.savez('init_params.npz', params=init_params)
            preds, labels, final_params = train(properties['num_epochs'], opt_init, opt_update, get_params, init_params)

        np.savez('final_params.npz', params=final_params)
        
    elif properties['run_type'] == 'test':
        # selecte the run_{}.npz file to grab parameters from
        if 'run_num' in properties:
            testing_params = np.load('run_{}.npz'.format(properties['run_num']))['params']
        else:
            _, testing_params = initialize_parameters(None)
            
        losses = run_test(properties['num_epochs'],testing_params)
