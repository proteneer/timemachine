from typing import Optional

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS
from scipy.spatial.distance import cdist

from timemachine.fe.topology import AtomMappingError


class CompareDist(rdFMCS.MCSAtomCompare):
    """Custom atom comparison: use positions within generated conformer"""

    def __init__(self, threshold=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def compare(self, p, mol1, atom1, mol2, atom2):
        """Atoms match if within 0.5 Å

        Signature from super method:
        (MCSAtomCompareParameters)parameters, (Mol)mol1, (int)atom1, (Mol)mol2, (int)atom2) -> bool
        """
        x_i = mol1.GetConformer(0).GetPositions()[atom1]
        x_j = mol2.GetConformer(0).GetPositions()[atom2]
        return bool(np.linalg.norm(x_i - x_j) <= self.threshold)  # must convert from np.bool_ to Python bool!


class CompareDistNonterminal(rdFMCS.MCSAtomCompare):
    """
    Custom comparator used in the FMCS code.
    This allows two atoms to match if:
        1. Neither atom is a terminal atom (H, F, Cl, Halogens etc.)
        2. They are within 1 angstrom of each other.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compare(self, p, mol1, atom1, mol2, atom2):

        if mol1.GetAtomWithIdx(atom1).GetDegree() == 1:
            return False
        if mol2.GetAtomWithIdx(atom2).GetDegree() == 1:
            return False

        x_i = mol1.GetConformer(0).GetPositions()[atom1]
        x_j = mol2.GetConformer(0).GetPositions()[atom2]
        if np.linalg.norm(x_i - x_j) > 1.0:  # angstroms
            return False
        else:
            return True


def mcs_map(a, b, threshold: float = 2.0, timeout: int = 5, smarts: Optional[str] = None):
    """Find the MCS map of going from A to B"""
    params = rdFMCS.MCSParameters()
    params.BondCompareParameters.CompleteRingsOnly = 1
    params.BondCompareParameters.RingMatchesRingOnly = 1
    params.BondTyper = rdFMCS.BondCompare.CompareAny

    params.AtomCompareParameters.CompleteRingsOnly = 1
    params.AtomCompareParameters.RingMatchesRingOnly = 1
    params.AtomCompareParameters.matchValences = 0
    params.AtomCompareParameters.MaxDistance = threshold
    params.AtomTyper = rdFMCS.AtomCompare.CompareAny

    params.Timeout = timeout
    if smarts is not None:
        params.InitialSeed = smarts
    return rdFMCS.FindMCS([a, b], params)


def mcs_map_graph_only_complete_rings(a, b, timeout: int = 3600, smarts: Optional[str] = None):
    """Find the MCS map of going from A to B, disregarding conformer information. This also ensures
    that core-core bonds are not broken."""
    return rdFMCS.FindMCS(
        [a, b],
        maximizeBonds=True,
        timeout=timeout,
        matchValences=False,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        matchChiralTag=False,
        atomCompare=Chem.rdFMCS.AtomCompare.CompareAny,
        bondCompare=Chem.rdFMCS.BondCompare.CompareAny,
    )


# 4. Construct and serialize the relative transformations
def _check_core_map_distances(mol_a, mol_b, core, threshold=0.5) -> bool:
    """compute vector of distances[i] = distance(conf_a[core_a[i]], conf_b[core_b[i]]),
    check whether distances[i] <= threshold for all i"""

    a, b = core[:, 0], core[:, 1]
    conf_a = mol_a.GetConformer(0).GetPositions()
    conf_b = mol_b.GetConformer(0).GetPositions()
    distances = np.linalg.norm(conf_a[a] - conf_b[b], axis=1)
    return (distances <= threshold).all()


def get_core_by_mcs(mol_a, mol_b, query, threshold=0.5):
    """Return np integer array that can be passed to RelativeFreeEnergy constructor

    Parameters
    ----------
    mol_a, mol_b, query : RDKit molecules
    threshold : float, in angstroms

    Returns
    -------
    core : np.ndarray of ints, shape (n_MCS, 2)

    Notes
    -----
    * Warning! Some atoms that intuitively should be mapped together are not,
        when threshold=0.5 Å in custom atom comparison, because conformers aren't
        quite aligned enough.
    * Warning! Because of the intermediate representation of a substructure query,
        the core indices can get flipped around,
        for example if the substructure match hits only part of an aromatic ring.

        In some cases, this means that pairs of atoms that do not satisfy the
        atom comparison function can be mapped together.
    """

    # fetch conformer, assumed aligned
    conf_a = mol_a.GetConformer(0).GetPositions()
    conf_b = mol_b.GetConformer(0).GetPositions()

    # note that >1 match possible here -- must pick minimum-cost match
    matches_a = mol_a.GetSubstructMatches(query, uniquify=False)
    matches_b = mol_b.GetSubstructMatches(query, uniquify=False)

    cost = np.zeros((len(matches_a), len(matches_b)))
    for i, a in enumerate(matches_a):
        for j, b in enumerate(matches_b):
            # if a single pair is outside of threshold, we set the cost to inf
            dij = np.linalg.norm(conf_a[np.array(a)] - conf_b[np.array(b)], axis=1)
            cost[i, j] = dij.sum() if np.all(dij < threshold) else np.inf

    # find (i,j) = argmin cost
    min_i, min_j = np.unravel_index(np.argmin(cost, axis=None), cost.shape)
    # print(f'argmin of {n_a} x {n_b} cost matrix: {(min_i, min_j)} ')
    # TODO: maybe also print the difference between min(cost) and cost[0,0],
    #   to see how big of a difference it made to pick the default

    # TODO: is there a way to use the matching from MCS directly?

    # concatenate into (n_atoms, 2) array
    inds_a, inds_b = matches_a[min_i], matches_b[min_j]
    core = np.array([inds_a, inds_b]).T

    if not _check_core_map_distances(mol_a, mol_b, core, threshold):
        raise AtomMappingError(f"not all mapped atoms are within {threshold:.3f}Å of each other")

    return core


def _assert_core_reasonableness(mol_a, mol_b, core):
    # TODO move any useful run-time assertions from this script into tests/

    # bounds
    assert max(core[:, 0]) < mol_a.GetNumAtoms()
    assert max(core[:, 1]) < mol_b.GetNumAtoms()

    # uniqueness
    assert len(set(core[:, 0])) == len(core)
    assert len(set(core[:, 1])) == len(core)


def get_core_by_geometry(mol_a, mol_b, threshold=0.5):
    """Only allow to map a pair of atoms together if their conformer coordinates are within threshold.

    Of the allowable core mappings, return the one that contains only atom pairs (i, j)
    where i in mol_a has exactly one neighbor j in mol_b within threshold
    """
    core = simple_geometry_mapping(mol_a, mol_b, threshold)
    _assert_core_reasonableness(mol_a, mol_b, core)
    return core


def _weighted_adjacency_graph(conf_a, conf_b, threshold=1.0):
    """construct a networkx graph with
    nodes for atoms in conf_a, conf_b, and
    weighted edges connecting (conf_a[i], conf_b[j])
        if distance(conf_a[i], conf_b[j]) <= threshold,
        with weight = threshold - distance(conf_a[i], conf_b[j])
    """
    distances = cdist(conf_a, conf_b)
    within_threshold = distances <= threshold

    g = nx.Graph()
    for i in range(len(within_threshold)):
        neighbors_of_i = np.where(within_threshold[i])[0]
        for j in neighbors_of_i:
            g.add_edge(f"conf_a[{i}]", f"conf_b[{j}]", weight=threshold - distances[i, j])
    return g


def _core_from_matching(matching):
    """matching is a set of pairs of node names"""

    # 'conf_b[9]' -> 9
    ind_from_node_name = lambda name: int(name.split("[")[1].split("]")[0])

    match_list = list(matching)

    inds_a = [ind_from_node_name(u) for (u, _) in match_list]
    inds_b = [ind_from_node_name(v) for (_, v) in match_list]

    return np.array([inds_a, inds_b]).T


def core_from_distances(mol_a, mol_b, threshold=1.0):
    """
    TODO: docstring
    TODO: test
    """
    # fetch conformer, assumed aligned
    conf_a = mol_a.GetConformer(0).GetPositions()
    conf_b = mol_b.GetConformer(0).GetPositions()

    g = _weighted_adjacency_graph(conf_a, conf_b, threshold)

    matching = nx.algorithms.matching.max_weight_matching(g, maxcardinality=True)

    return _core_from_matching(matching)


def simple_geometry_mapping(mol_a, mol_b, threshold=0.5):
    """For each atom i in conf_a, if there is exactly one atom j in conf_b
    such that distance(i, j) <= threshold, add (i,j) to atom mapping

    Notes
    -----
    * Warning! There are many situations where a pair of atoms that shouldn't be mapped together
        could appear within distance threshold of each other in their respective conformers
    """

    # fetch conformer, assumed aligned
    conf_a = mol_a.GetConformer(0).GetPositions()
    conf_b = mol_b.GetConformer(0).GetPositions()
    # TODO: perform initial alignment

    within_threshold = cdist(conf_a, conf_b) <= threshold
    num_neighbors = within_threshold.sum(1)
    num_mappings_possible = np.prod(num_neighbors[num_neighbors > 0])

    if max(num_neighbors) > 1:
        print(
            f"Warning! Multiple (~ {num_mappings_possible}) atom-mappings would be possible at threshold={threshold}Å."
        )
        print(f"Only mapping atoms that have exactly one neighbor within {threshold}Å.")
        # TODO: print more information about difference between size of set returned and set possible
        # TODO: also assert that only pairs of the same element will be mapped together

    inds = []
    for i in range(len(conf_a)):
        if num_neighbors[i] == 1:
            inds.append((i, np.argmax(within_threshold[i])))
    core = np.array(inds)
    return core
