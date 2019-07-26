import os
import io
import sys
import numpy as np
import time
import argparse

from rdkit import Chem
from rdkit.Chem import AllChem
from argparse import RawTextHelpFormatter
from tqdm import tqdm

from system import serialize
from system import forcefield
from system import simulation

from openforcefield.typing.engines.smirnoff import ForceField
from timemachine.observables import rmsd
from timemachine.lib import custom_ops

import multiprocessing

import traceback

import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def create_data(params):
    signal.signal(signal.SIGALRM, timeout_handler)

    filename,directory,idx = params
    
    process = multiprocessing.current_process()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(idx % 8)
    
    smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")
    
    # suppress output of functions
#     text_trap = io.StringIO()
#     sys.stdout = text_trap
    
    # Time out after 60 seconds
#     signal.alarm(60)
    
    try:
        if '.mol2' not in filename:
            raise Exception('not mol2')
        with open(os.path.join(directory,filename),'r') as f:
            structure_file = f.read()
        if ' H ' not in structure_file:
            raise Exception('no hydrogens')
        ref_mol = Chem.MolFromMol2Block(structure_file, sanitize=False, removeHs=False, cleanupSubstructures=True)
        Chem.SanitizeMol(ref_mol)
        AllChem.EmbedMolecule(ref_mol, clearConfs=True, useExpTorsionAnglePrefs=False, useBasicKnowledge=False)
        if '.' in Chem.MolToSmiles(ref_mol):
            raise Exception('multiple compounds for {}'.format(filename))
        if ref_mol.GetNumAtoms() > 36 or ref_mol.GetNumAtoms() < 8:
            raise Exception('num atoms')
        guest_potentials, smirnoff_params, smirnoff_param_groups, guest_conf, guest_masses = forcefield.parameterize(ref_mol, smirnoff)
        c = ref_mol.GetConformer(0)
        conf = np.array(c.GetPositions(), dtype=np.float64)
        guest_conf = conf/10 # convert to MD units
        guest_dp_idxs = np.array([0])
#         RG = simulation.run_simulation(
#             guest_potentials,
#             smirnoff_params,
#             smirnoff_param_groups,
#             guest_conf,
#             guest_masses,
#             guest_dp_idxs,
#             1
#         )
#         loss = rmsd.opt_rot_rmsd(guest_conf,RG[0][-1])
#         print('{} success'.format(filename))
        return [filename,guest_conf]
    except Exception as e:
        pass
        traceback.print_exc()
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='''Clean up structure data according to the following criteria:
- mol2 does not contain multiple molecules
- 8 < number of atoms < 36
- mol2 includes explicit hydrogens
- mol2 can be successfully parameterized
- mol2 can be successfully minimized
- RDKit can successfully generate a conformer

Writes a file training_data.npz that contains the filename and coordinates of each mol2 file that fits the criteria''',formatter_class=RawTextHelpFormatter)
    
    parser.add_argument('mol2_directory',metavar='dir',help='The directory containing mol2 files to sort')
    args = parser.parse_args()
    
    training_data = []
    mol2_dir = args.mol2_directory
    
    filenames = []
    
    batch_size = 64
    
    for idx,filename in enumerate(os.listdir(mol2_dir)):
        filenames.append(filename)
        
    p = multiprocessing.Pool(batch_size)
    num_data_points = len(filenames)
    num_batches = int(np.ceil(num_data_points/batch_size))
    
    start = time.time()
    
    results = []
    
    for b_idx in tqdm(range(num_batches)):
        start_idx = b_idx*batch_size
        end_idx = min((b_idx+1)*batch_size, num_data_points)
        batch_data = filenames[start_idx:end_idx]

        args = []

        for b_idx, b in enumerate(batch_data):
            args.append([b,mol2_dir,b_idx])

        batch_res = p.map(create_data,args)
        
        for item in batch_res:
            if item is not None:
                results.append(item)
    
    print(time.time() - start)
    print(len(results))

    np.savez('training_data.npz',data=results)