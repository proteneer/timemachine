# Construct a star map for the fep-benchmark hif2a ligands

from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS, Draw

import matplotlib.pyplot as plt

root = Path(__file__).parent.parent.parent

# 0. Get force field
from ff.handlers.deserialize import deserialize_handlers

path_to_ff = str(root.joinpath('ff/params/smirnoff_1_1_0_ccc.py'))
with open(path_to_ff) as f:
    ff_handlers = deserialize_handlers(f.read())

# 1. Get ligands
# TODO: possibly Git submodule to fep-benchmark inside datasets/, rather than copying fep-benchmark into datasets

path_to_ligands = str(root.joinpath('datasets/fep-benchmark/hif2a/ligands.sdf'))

supplier = Chem.SDMolSupplier(path_to_ligands)
mols = []
for mol in supplier:
    mols.append(mol)

# 2. Identify ligand subset that shares a common substructure

# SMARTS courtesy of YTZ (Jan 26, 2021)
eyeballed_smarts_pattern = '[*]~1~[*]~[*]~2~[*]~[*]~[*]~[*](~[#8]~[*]~3~[*]~[*]~[*]~[*]~[*]~3)~[*]~2~[*]~1'
bicyclic_query_mol = Chem.MolFromSmarts(eyeballed_smarts_pattern)

# filter matches
mols_with_core_1 = [mol for mol in mols if mol.HasSubstructMatch(bicyclic_query_mol)]


# TODO: save this image...
# Draw.MolsToGridImage(mols_with_core_1, molsPerRow=5, subImgSize=(200,200))

# 3. Identify the center or "hub" of the star map

# 3.1. Define a custom maximum common substructure search method

class CompareDist(rdFMCS.MCSAtomCompare):
    """Custom atom comparison: use positions within generated conformer"""

    def compare(self, p, mol1, atom1, mol2, atom2):
        """Atoms match if within 0.5 Å

        Signature from super method:
        (MCSAtomCompareParameters)parameters, (Mol)mol1, (int)atom1, (Mol)mol2, (int)atom2) -> bool
        """
        x_i = mol1.GetConformer(0).GetPositions()[atom1]
        x_j = mol2.GetConformer(0).GetPositions()[atom2]
        return bool(np.linalg.norm(x_i - x_j) <= 0.5)  # must convert from np.bool_ to Python bool!


def mcs_map(a, b):
    """Find the MCS map of going from A to B"""
    params = rdFMCS.MCSParameters()
    params.AtomTyper = CompareDist()
    return rdFMCS.FindMCS([a, b], params)


# 3.2. Relate MCS size to transformation size

def transformation_size(n_A, n_B, n_MCS):
    """Heuristic size of transformation in terms of
	the number of atoms in A, B, and MCS(A, B)

    Notes
    -----
    * YTZ suggests (2021/01/26)
        (n_A + n_B) - n_MCS
	which is size-extensive

    * JF modified (2021/01/27)
        (n_A - n_MCS) + (n_B - n_MCS)
	which is 0 when A = B

    * Alternatives considered but not tried:
        * min(n_A, n_B) - n_MCS
        * max(n_A, n_B) - n_MCS
        * 1.0 - (n_MCS / min(n_A, n_B))

    TODO: move this into a utility module or the free energy module
    """
    return (n_A + n_B) - 2 * n_MCS


# 3.3. Identify the molecule i for which the total amount of RelativeFreeEnergy estimation effort is smallest
# (argmin_i \sum_j transformation_size(mols[i], mols[j]))

def compute_all_pairs_mcs(mols):
    """Generate square matrix of MCS(mols[i], mols[j]) for all pairs i,j

    Parameters
    ----------
    mols : iterable of RDKit Mols

    Returns
    -------
    mcs_s : numpy.ndarray of rdkit.Chem.rdFMCS.MCSResult's, of shape (len(mols), len(mols))

    Notes
    -----
    TODO: mcs_map should be symmetric, so only have to do the upper triangle of this matrix
    """
    mcs_s = np.zeros((len(mols), len(mols)), dtype=object)

    for i in range(len(mols)):
        for j in range(len(mols)):
            mcs_s[i, j] = mcs_map(mols[i], mols[j])
    return mcs_s


def compute_transformation_size_matrix(mols, mcs_s):
    N = len(mols)
    transformation_sizes = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            n_i = mols[i].GetNumAtoms()
            n_j = mols[j].GetNumAtoms()
            n_mcs = mcs_s[i, j].numAtoms
            transformation_sizes[i, j] = transformation_size(n_i, n_j, n_mcs)
    return transformation_sizes


mcs_s = compute_all_pairs_mcs(mols_with_core_1)
transformation_sizes = compute_transformation_size_matrix(mols_with_core_1, mcs_s)


def identify_hub(transformation_sizes):
    return int(np.argmin(transformation_sizes.sum(0)))


hub_index = identify_hub(transformation_sizes)
hub = mols_with_core_1[hub_index]


def plot_transformation_sizes(transformation_sizes):
    plt.imshow(transformation_sizes)
    plt.xlabel('molecule index')
    plt.ylabel('molecule index')
    plt.title('"size of transformation"\n$(n_A - n_{MCS}) + (n_B - n_{MCS})$')

    plt.tight_layout()

    plt.colorbar()


plot_transformation_sizes(transformation_sizes)
plt.savefig('transformation_sizes.png', bbox_inches='tight')


# 4. Construct and serialize the relative transformations
def get_core(mol_a, mol_b, query):
    """Return np integer array that can be passed to RelativeFreeEnergy constructor

    Parameters
    ----------
    mol_a, mol_b, query : RDKit molecules

    Returns
    -------
    core : np.ndarray of ints, shape (n_MCS, 2)

    TODO: move this into a utility module or the free energy module
    """
    inds_a = mol_a.GetSubstructMatch(query)
    inds_b = mol_b.GetSubstructMatch(query)
    core = np.array([inds_a, inds_b]).T
    return core


# for each "spoke" in the star map, construct serializable transformation "hub -> spoke"
others = list(mols_with_core_1)
others.pop(hub_index)

from fe.free_energy import RelativeFreeEnergy
from fe.topology import AtomMappingError

def get_mol_id(mol):
    return mol.GetPropsAsDict()['ID']

transformations = []
for spoke in others:
    core = get_core(hub, spoke, mcs_map(hub, spoke).queryMol)
    try:
        rfe = RelativeFreeEnergy(hub, spoke, core, ff_handlers)
        transformations.append(rfe)
    except AtomMappingError as e:
        print(f'atom mapping error in transformation {get_mol_id(hub)} -> {get_mol_id(spoke)}!')
        print(core)
        print(e)

# serialize
from pickle import dump

with open('relative_transformations.pkl', 'wb') as f:
    dump(transformations, f)
