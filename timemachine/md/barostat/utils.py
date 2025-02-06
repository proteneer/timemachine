import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist

from timemachine.potentials import HarmonicBond


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


def get_bond_list(harmonic_bond_potential: HarmonicBond) -> list[tuple[int, int]]:
    """Read off topology from indices of harmonic bond force

    Notes
    -----
    * Assumes all valence bonds are represented by this harmonic bond force.
        This assumption could break if there are multiple harmonic bond forces in the system,
        or if there are valence bonds not represented as harmonic bonds (e.g. as length constraints)
    """

    bond_list = [(i, j) for i, j in harmonic_bond_potential.idxs]
    return bond_list


def get_group_indices(bond_list: list[tuple[int, int]], num_atoms: int) -> list[NDArray]:
    """Connected components of bond graph"""

    topology = nx.Graph(bond_list)
    components = [np.array(list(sorted(c))) for c in nx.algorithms.connected_components(topology)]
    for i in range(len(components)):
        assert np.all(np.diff(components[i]) == 1)

    found_set = set()
    for grp in components:
        for idx in grp:
            assert idx < num_atoms
            found_set.add(idx)

    for atom_idx in range(num_atoms):
        if atom_idx not in found_set:
            components.append(np.array([atom_idx], dtype=np.int32))

    return components


def compute_intramolecular_distances(coords: NDArray, group_indices: list[NDArray]) -> list[NDArray]:
    """pairwise distances within each group"""
    return [pdist(coords[inds]) for inds in group_indices]
