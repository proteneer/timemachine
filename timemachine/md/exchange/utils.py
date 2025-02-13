import numpy as np
from rdkit import Chem

from timemachine.fe.utils import get_romol_conf
from timemachine.potentials.jax_utils import pairwise_distances


def get_radius_of_mol_pair(mol_a: Chem.Mol, mol_b: Chem.Mol) -> float:
    """Takes two molecules, computes the max pairwise distance within the molecule coordinates,
    treating that as a diameter and returns the radius
    """
    conf = np.concatenate([get_romol_conf(mol_a), get_romol_conf(mol_b)])
    diameter = np.max(pairwise_distances(conf))
    return diameter / 2
