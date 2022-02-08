from typing import List, Tuple

import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist

from timemachine.lib.potentials import HarmonicBond


def compute_box_volume(box: np.ndarray) -> float:
    assert box.shape == (3, 3)
    return np.linalg.det(box)


def compute_box_center(box: np.ndarray) -> np.ndarray:
    # assume axis-aligned box (nothing off diagonal)
    assert box.shape == (3, 3)
    assert np.linalg.norm(box - np.diag(np.diag(box))) == 0

    center = np.sum(box / 2, axis=0)

    assert center.shape == (3,)

    return center


def get_bond_list(harmonic_bond_potential: HarmonicBond) -> List[Tuple[int, int]]:
    """Read off topology from indices of harmonic bond force

    Notes
    -----
    * Assumes all valence bonds are represented by this harmonic bond force.
        This assumption could break if there are multiple harmonic bond forces in the system,
        or if there are valence bonds not represented as harmonic bonds (e.g. as length constraints)
    """

    bond_list = list(map(tuple, harmonic_bond_potential.get_idxs()))
    return bond_list


def get_group_indices(bond_list: List[np.array]) -> List[np.array]:
    """Connected components of bond graph"""

    topology = nx.Graph(bond_list)
    components = [np.array(list(c)) for c in nx.algorithms.connected_components(topology)]
    return components


def compute_intramolecular_distances(coords: np.array, group_indices: List[np.array]) -> List[np.array]:
    """pairwise distances within each group"""
    return [pdist(coords[inds]) for inds in group_indices]
