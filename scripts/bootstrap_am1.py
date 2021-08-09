import argparse
import numpy as np
import os
import pickle

from rdkit import Chem
from openeye import oechem
from openeye import oequacpac
from openeye import oeomega
from ff.handlers.nonbonded import convert_to_oe

ONE_4PI_EPS0 = 138.935456

from multiprocessing import Pool

all_patterns = [
    '[#1:1]',
    '[#1:1]-[#6X4]',
    '[#1:1]-[#6X4]-[#7,#8,#9,#16,#17,#35]',
    '[#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]',
    '[#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])(-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]',
    '[#1:1]-[#6X4]~[*+1,*+2]',
    '[#1:1]-[#6X3]',
    '[#1:1]-[#6X3]~[#7,#8,#9,#16,#17,#35]',
    '[#1:1]-[#6X3](~[#7,#8,#9,#16,#17,#35])~[#7,#8,#9,#16,#17,#35]',
    '[#1:1]-[#6X2]',
    '[#1:1]-[#7]',
    '[#1:1]-[#8]',
    '[#1:1]-[#16]',
    '[#6:1]',
    '[#6X2:1]',
    '[#6X4:1]',
    '[#6:1]~[#7]',
    '[#6:1]-[#7]',
    '[#6X4:1]-[#7]',
    '[#6X3:1]-[#7]',
    '[#6X3:1](-[#6X3])-[#7]',
    '[#6X3:1](-[#6X3])-[#7X3]',
    '[#6X3:1](-[#7])-[#7]',
    '[#6X3:1](-[#8])-[#7]',
    '[#6:1]:[#7]',
    '[#6X3:1]:[#7]',
    '[#6X3:1](-[#7]):[#7]',
    '[#6X3:1](-[#8]):[#7]',
    '[#6:1]-[#9]',
    '[#6X4:1]-[#9]',
    '[#6X4:1](-[#6X3])-[#9]',
    '[#6X4:1](-[#7])-[#9]',
    '[#6X4:1](-[#8])-[#9]',
    '[#6X3:1]-[#9]',
    '[#6:1]~[#16]',
    '[#6:1]-[#16]',
    '[#6:1](-[#6X3])-[#16]',
    '[#6X3:1](-[#6X3])-[#16]',
    '[#6X3:1](-[#6H1])-[#16]',
    '[#6X3:1](-[#6X3])-[#16](-[#6H0])',
    '[#6X3:1](-[#6X3])-[#16](-[#6H1])',
    '[#6:1](-[#7X4])-[#16]',
    '[#6:1](-[#7X3])-[#16]',
    '[#6X3:1](-[#7X3])-[#16](~[#6])',
    '[#6:1]=[#16]',
    '[#6:1]:[#16]',
    '[#6:1](=[#8])-[#16]',
    '[#6:1]-[#16]=[#8]',
    '[#6:1](-[#16])-[#9]',
    '[#6:1](-[#16])-[#17]',
    '[#6:1](-[#16])-[#35]',
    '[#8:1]',
    '[#8X2H0+0:1]',
    '[#8X2H1+0:1]',
    '[#7:1]',
    '[#7X3:1]',
    '[#7X4:1]',
    '[#7:1]~[#7]',
    '[#7:1]:[#7]',
    '[#7:1]=[#7]',
    '[#7:1]-[#7]',
    '[#7X3:1]-[#6X3]',
    '[#7X3:1]-[#6X3](-[#6X3])',
    '[#7X3:1](-[#6X3])-[#6X3](-[#6X3])',
    '[#7H1:1](-[#6X3])-[#6X3](-[#6X3])',
    '[#7X3:1]-[#6X3](-[#7])',
    '[#7X3:1]-[#6X3](-[#8])',
    '[#7X3:1]-[#7X2]',
    '[#7:1]-[#16]',
    '[#7:1](-[#6])-[#16]',
    '[#7X3:1]-[#16]',
    '[#7H1:1]-[#7]',
    '[#7H0:1]=[#7]',
    '[#16:1]',
    '[#16:1]-[#6]',
    '[#16:1]~[#6]',
    '[#16:1]~[#6]~[#7]',
    '[#16:1]~[#7]',
    '[#16:1]=[#8]',
    '[#15:1]',
    '[#9:1]',
    '[#17:1]',
    '[#35:1]',
    '[#53:1]',
    '[#3+1:1]',
    '[#11+1:1]',
    '[#19+1:1]',
    '[#37+1:1]',
    '[#55+1:1]',
    '[#9X0-1:1]',
    '[#17X0-1:1]',
    '[#35X0-1:1]',
    '[#53X0-1:1]'
]


def match_smirks(mol, smirks):
    
    # Make a copy of the molecule
    rdmol = Chem.Mol(mol)
    # Use designated aromaticity model
    Chem.SanitizeMol(rdmol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY)
    Chem.SetAromaticity(rdmol, Chem.AromaticityModel.AROMATICITY_MDL)
    
    # Set up query.
    qmol = Chem.MolFromSmarts(smirks)  #cannot catch the error
    if qmol is None:
        raise ValueError('RDKit could not parse the SMIRKS string "{}"'.format(smirks))

    # Create atom mapping for query molecule
    idx_map = dict()
    for atom in qmol.GetAtoms():
        smirks_index = atom.GetAtomMapNum()
        if smirks_index != 0:
            idx_map[smirks_index - 1] = atom.GetIdx()
    map_list = [idx_map[x] for x in sorted(idx_map)]

    # Perform matching
    matches = list()
    for match in rdmol.GetSubstructMatches(qmol, uniquify=False):
        mas = [match[x] for x in map_list]
        matches.append(tuple(mas))

    return matches


def parameterize_mol(ff, mol):

    param_idxs = np.zeros(mol.GetNumAtoms(), dtype=np.int32)
    for pattern_idx, pattern in enumerate(ff):
        matches = match_smirks(mol, pattern)
        for m in matches:
            param_idxs[m] = pattern_idx

    return param_idxs

def loss(q1, q2):
    q1 = np.array(q1)
    q2 = np.array(q2)

    print(q1, q2)
    return np.mean(np.abs(q1-q2))

# sum is the score of the *variance* multiplied by the number of patterns
def score(stds):
    variance = np.power(stds, 2)
    return np.sum(variance)*len(stds)


def prep_structure(rdmol):
    oemol = convert_to_oe(rdmol)
    omega = oeomega.OEOmega()
    # omega.SetIncludeInput(True)
    omega.SetMaxSearchTime(30)
    omega.SetCanonOrder(False)
    omega.SetSampleHydrogens(True)
    eWindow = 15.0
    omega.SetEnergyWindow(eWindow)
    # omega.SetMaxConfs(800)
    omega.SetMaxConfs(400)
    omega.SetRMSThreshold(1.0)

    if omega(oemol):
        result = oequacpac.OEAssignCharges(oemol, oequacpac.OEAM1BCCELF10Charges())
        if result is False:
            return None
        else:
            charges = []
            for index, atom in enumerate(oemol.GetAtoms()):
                q = atom.GetPartialCharge()*np.sqrt(ONE_4PI_EPS0)
                charges.append(q)
            return charges
    else:
        return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Quick Test')
    parser.add_argument('--ligand_sdf', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--pool_size', type=int, required=True)
    parser.add_argument('--limit', type=int, required=True)
    args = parser.parse_args()

    chkpt_file = args.checkpoint

    if os.path.exists(chkpt_file):
        all_rdmols, am1_charges = pickle.load(open(chkpt_file, "rb"))
    else:
        pool = Pool(args.pool_size)
        suppl = Chem.SDMolSupplier(args.ligand_sdf, removeHs=False)
        all_rdmols = []
        for mol_idx, mol in enumerate(suppl):

            if mol_idx > args.limit:
                break

            all_rdmols.append(Chem.AddHs(mol))
        
        print("Total of", len(all_rdmols), "mols")

        am1_charges = pool.map(prep_structure, all_rdmols)
        pool.close()

        pickle.dump([all_rdmols, am1_charges], open(chkpt_file, "wb"))

    ff = all_patterns

    all_guest_mols = []
    param_vals = []
    for _ in ff:
        param_vals.append([])

    for idx, (guest_mol, true_q) in enumerate(zip(all_rdmols, am1_charges)):

        if true_q is not None:
            assert guest_mol.GetNumAtoms() == len(true_q)
            all_guest_mols.append(guest_mol)
            param_idxs = parameterize_mol(ff, guest_mol)

            for atom_idx, p in enumerate(param_idxs):
                am1_charge = true_q[atom_idx]
                param_vals[p].append(am1_charge)

    for p_idx, vals in enumerate(param_vals):

        q = np.mean(vals)
        if np.isnan(q):
            q = 0
        
        print("['%s',"%ff[p_idx], "{:6.4f}],".format(q/11.787089))