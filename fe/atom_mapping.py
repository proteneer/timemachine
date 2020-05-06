import numpy as np

import rdkit
from rdkit import Chem
from rdkit.Chem import rdFMCS

from timemachine.potentials import jax_utils

def compute_distance(mol_a, mol_b, a_to_b):
    conformer_a = mol_a.GetConformer(0)
    mol_a_conf = np.array(conformer_a.GetPositions(), dtype=np.float64)

    conformer_b = mol_b.GetConformer(0)
    mol_b_conf = np.array(conformer_b.GetPositions(), dtype=np.float64)

    i_idxs = list(a_to_b.keys())
    j_idxs = list(a_to_b.values())

    r_i = mol_a_conf[i_idxs]
    r_j = mol_b_conf[j_idxs]

    all_dists = jax_utils.distance(r_i, r_j, None)

    return all_dists

class CompareDist(rdFMCS.MCSAtomCompare):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compare(self, p, mol1, atom1, mol2, atom2):

        x_i = mol1.GetConformer(0).GetAtomPosition(atom1)
        x_j = mol2.GetConformer(0).GetAtomPosition(atom2)

        if np.linalg.norm(x_i-x_j) > 0.5:
            return False
        else:
            return True

def mcs_map(a, b):
    """
    Find the MCS map of going from A to B
    """
    params = rdFMCS.MCSParameters()
    params.AtomTyper = CompareDist()
    core_pattern = rdFMCS.FindMCS([a, b], params).smartsString

    # figure out ring stuff later
    # ringCompare=Chem.rdFMCS.RingCompare.StrictRingFusion).smartsString
    # ringCompare=Chem.rdFMCS.RingCompare.PermissiveRingFusion).smartsString
    # ringCompare=Chem.rdFMCS.RingCompare.IgnoreRingFusion).smartsString
    core = Chem.MolFromSmarts(core_pattern)

    a_to_core = {}
    core_to_b = {}

    a_matches = a.GetSubstructMatches(core, uniquify=False)
    b_matches = b.GetSubstructMatches(core, uniquify=False)

    mean_dists = []
    all_dists = []
    all_a_to_bs = []

    for a_match in a_matches:
        for b_match in b_matches:

            a_to_core = {}
            for core_idx, a_idx in enumerate(a_match):
                a_to_core[a_idx] = core_idx

            core_to_b = {}
            for core_idx, atom in enumerate(b_match):
                core_to_b[core_idx] = atom

            a_to_b = {}
            for a_idx, c_idx in a_to_core.items():
                a_to_b[a_idx] = core_to_b[a_to_core[a_idx]]

            # print("dist", compute_distance(a, b, a_to_b))
            dists = compute_distance(a, b, a_to_b)
            mean_dists.append(np.mean(dists))
            all_dists.append(dists)
            all_a_to_bs.append(a_to_b)


    min_mean_arg = np.argmin(mean_dists)
    for d in all_dists[min_mean_arg]:
        if d > 0.5:
            print("REALLY BAD WARNING: Matched core atoms that have a distance of", d)
            # assert 0

    return all_a_to_bs[min_mean_arg]


if __name__ == "__main__":

    a = Chem.MolFromSmiles("c1ccccc1F")
    b = Chem.MolFromSmiles("c1cc(F)ccc1F")
    a = Chem.AddHs(a)
    b = Chem.AddHs(b)
    a_to_b = mcs_map(a, b)
    print(a_to_b)