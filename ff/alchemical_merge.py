from rdkit import Chem
import numpy as np
from rdkit.Chem import rdFMCS
from ff import forcefield


def get_masses(m):
    masses = []
    for a in m.GetAtoms():
        masses.append(a.GetSymbol())
    return masses

off = forcefield.Forcefield("ff/smirnoff_1.1.0.py")

mol1 = Chem.AddHs(Chem.MolFromSmiles("c1ccc(O)cc1"))
mol2 = Chem.AddHs(Chem.MolFromSmiles("c1c(F)cccc1"))
m1_masses = get_masses(mol1)
m2_masses = get_masses(mol2)

core_pattern = rdFMCS.FindMCS([mol1, mol2]).smartsString
core = Chem.MolFromSmarts(core_pattern)

R_C = core.GetNumAtoms()
R_A = mol1.GetNumAtoms()-core.GetNumAtoms()
R_B = mol2.GetNumAtoms()-core.GetNumAtoms()


def reorder(mol, core, offset):
    # use 3d predicates to deal with symmetry/redundant matches later
    core_to_old = np.array(mol.GetSubstructMatches(core)[0])
    old_to_new = [None]*mol.GetNumAtoms()
    for core_idx, target_idx in enumerate(core_to_old):
        old_to_new[target_idx] = core_idx
    counter = core.GetNumAtoms()+offset
    for old_idx, elem in enumerate(old_to_new):
        if elem is None:
            old_to_new[old_idx] = counter
            counter += 1
    return old_to_new

# bonds, angles, torsions, lj, nb, exclusions, GB, will all need to use this new mapping
mol1_map = reorder(mol1, core, offset=0)
mol2_map = reorder(mol2, core, offset=R_A)

new_masses = [None]*(R_C+R_A+R_B)
for src, dst in enumerate(mol1_map):
    new_masses[dst] = m1_masses[src]
for src, dst in enumerate(mol2_map):
    new_masses[dst] = m2_masses[src]

print("combined masses", new_masses)

def order(src, dst):
    assert src != dst
    if src > dst:
        return dst, src
    return src, dst

def iterate_bonds(mol_bonds, mol_map):
    core_bonds = {}
    new_bond_idxs = []
    new_bond_params = []
    for (src, dst), param_idxs in zip(*mol_bonds):
        param_idxs = tuple(param_idxs.tolist())
        new_src, new_dst = mol_map[src], mol_map[dst]
        key = order(new_src, new_dst)
        if new_src < R_C and new_dst < R_C:
            core_bonds[key] = param_idxs
        else:
            new_bond_idxs.append(key)
            new_bond_params.append(param_idxs)

    return core_bonds, new_bond_idxs, new_bond_params

mol1_bonds = off.parameterize(mol1)['HarmonicBond']
mol1_core, mol1_new_bond_idxs, mol1_new_bond_params = iterate_bonds(mol1_bonds, mol1_map)

mol2_bonds = off.parameterize(mol2)['HarmonicBond']
mol2_core, mol2_new_bond_idxs, mol2_new_bond_params = iterate_bonds(mol2_bonds, mol2_map)


def core_check(core_a, core_b):
    """
    Check k,v in core_a against core_b for consistency. Mismatching bond idxs need to be
    alchemically modified and no longer defined as a core atom.
    """
    good_core = dict()
    new_bond_idxs = []
    new_bond_params = []
    for k, v in core_a.items():
        if v != core_b[k]:
            new_bond_idxs.append(k)
            new_bond_params.append(v)
        else:
            good_core[k] = v

    return good_core, new_bond_idxs, new_bond_params

gc1, b1, p1 = core_check(mol1_core, mol2_core)
gc2, b2, p2 = core_check(mol2_core, mol1_core)

# ensure the cores are consistent
assert gc1 == gc2
mol1_new_bond_idxs.extend(b1)
mol1_new_bond_params.extend(p1)

mol2_new_bond_idxs.extend(b2)
mol2_new_bond_params.extend(p2)

lambda_idxs = []
combined_bond_idxs = []
combined_bond_params = []



for k, v in gc1.items():
    lambda_idxs.append(0)
    combined_bond_idxs.append(k)
    combined_bond_params.append(v)
    
for k, v in zip(mol1_new_bond_idxs, mol1_new_bond_params):
    lambda_idxs.append(1)
    combined_bond_idxs.append(k)
    combined_bond_params.append(v)
    
for k, v in zip(mol2_new_bond_idxs, mol2_new_bond_params):
    lambda_idxs.append(-1)
    combined_bond_idxs.append(k)
    combined_bond_params.append(v)

for l_idx, bond_idxs, bond_params in zip(lambda_idxs, combined_bond_idxs, combined_bond_params):
    print(l_idx, new_masses[bond_idxs[0]], new_masses[bond_idxs[1]], bond_params)    

    # print(lambda_idxs)
    # print(combined_bond_idxs)
    # print(combined_bond_params)