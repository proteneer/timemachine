from collections import defaultdict
from functools import partial
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

from timemachine.fe import mcgregor
from timemachine.fe.chiral_utils import ChiralRestrIdxSet, has_chiral_atom_flips, setup_find_flipped_planar_torsions
from timemachine.fe.single_topology import AtomMapFlags, AtomMapMixin, make_find_chirally_valid_dummy_groups
from timemachine.fe.utils import get_romol_bonds, get_romol_conf

# (ytz): Just like how one should never re-write an MD engine, one should never rewrite an MCS library.
# Unless you have to. And now we have to. If you want to understand what this code is doing, it
# is strongly recommended that the reader reads:

# Backtrack search algorithms and the maximal common subgraph problem
# James J McGregor,  January 1982, https://doi.org/10.1002/spe.4380120103

# Theoretical Tricks
# ------------------
# Historically, MCS methods have relied on finding the largest core as soon as possible. However, this can pose difficulties
# since we may get stuck in a local region of poor quality (that end up having far smaller than the optimal). Our algorithm
# has several clever tricks up its sleeve in that we:

# - designed the method for free energy methods where the provided two molecules are aligned.
# - refine the row/cols of marcs (edge-edge mapping matrix) when a new atom-atom mapping is proposed
# - prune by looking at maximum number of row edges and column edges, i.e. arcs_left min(max_row_edges, max_col_edges)
# - only generate an atom-mapping between two mols, whereas RDKit generates a common substructure between N mols
# - operate on anonymous graphs whose atom-atom compatibility depends on a predicates matrix, such that a 1 is when
#   if atom i in mol_a is compatible with atom j in mol_b, and 0 otherwise. We do not implement a bond-bond compatibility matrix.
# - allow for the generation of disconnected atom-mappings, which is very useful for linker changes etc.
# - re-order the vertices in graph based on the degree, this penalizes None mapping by the degree of the vertex
# - provide a hard guarantee for timeout, i.e. completion of the algorithm implies global optimum(s) have been found
# - when searching for atoms in mol_b to map, we prioritize based on distance
# - runs the recursive algorithm in iterations with thresholds, which avoids us getting stuck in a branch with a low
#   max_num_edges. we've seen cases where we get stuck in an edge size of 45 but optimal edge mapping has 52 edges.
# - termination guarantees correctness. otherwise an assertion is thrown since the distance (in terms of # of edges mapped)
#   is unknown relative to optimal.

# Engineering Tricks
# ------------------
# This is entirely written in python, which lends to its ease of use and modifiability. The following optimizations were
# implemented (without changing the number of nodes visited):
# - multiple representations of graph structures to improve efficiency
# - refinement of marcs matrix is done on uint8 arrays


def get_cores_and_diagnostics(
    mol_a,
    mol_b,
    ring_cutoff,
    chain_cutoff,
    max_visits,
    max_connected_components: Optional[int],
    min_connected_component_size: int,
    max_cores,
    enforce_core_core,
    ring_matches_ring_only,
    enforce_chiral,
    disallow_planar_torsion_flips,
    min_threshold,
    initial_mapping,
    enforce_chirally_valid_dummy_groups: bool,
) -> Tuple[List[NDArray], mcgregor.MCSDiagnostics]:
    """Same as :py:func:`get_cores`, but additionally returns diagnostics collected during the MCS search."""
    assert max_cores > 0

    get_cores_ = partial(
        _get_cores_impl,
        ring_cutoff=ring_cutoff,
        chain_cutoff=chain_cutoff,
        max_visits=max_visits,
        max_connected_components=max_connected_components,
        min_connected_component_size=min_connected_component_size,
        max_cores=max_cores,
        enforce_core_core=enforce_core_core,
        ring_matches_ring_only=ring_matches_ring_only,
        enforce_chiral=enforce_chiral,
        disallow_planar_torsion_flips=disallow_planar_torsion_flips,
        min_threshold=min_threshold,
        enforce_chirally_valid_dummy_groups=enforce_chirally_valid_dummy_groups,
    )

    # we require that mol_a.GetNumAtoms() <= mol_b.GetNumAtoms()
    if mol_a.GetNumAtoms() > mol_b.GetNumAtoms():
        # reverse the columns of initial_mapping and the resulting cores
        initial_mapping_r = initial_mapping[:, ::-1] if initial_mapping is not None else None
        all_cores_r, mcs_diagnostics = get_cores_(mol_b, mol_a, initial_mapping=initial_mapping_r)
        all_cores = [core_r[:, ::-1] for core_r in all_cores_r]
    else:
        all_cores, mcs_diagnostics = get_cores_(mol_a, mol_b, initial_mapping=initial_mapping)
    return all_cores, mcs_diagnostics


def get_cores(
    mol_a,
    mol_b,
    ring_cutoff,
    chain_cutoff,
    max_visits,
    max_connected_components: Optional[int],
    min_connected_component_size: int,
    max_cores,
    enforce_core_core,
    ring_matches_ring_only,
    enforce_chiral,
    disallow_planar_torsion_flips,
    min_threshold,
    initial_mapping,
    enforce_chirally_valid_dummy_groups: bool,
) -> List[NDArray]:
    """
    Finds set of cores between two molecules that maximizes the number of common edges.

    If either atom i or atom j is in a ring then the dist(i,j) < ring_cutoff, otherwise dist(i,j) < chain_cutoff

    Additional notes
    ----------------
    1) The returned cores are jointly sorted in increasing order based on the number of core-dummy bonds broken,
       the sum of valence values changed, and the mean square distance of the alignment.
    2) The number of cores atoms may vary slightly, but the number of mapped edges are the same.
    3) If a time-out has occurred due to max_visits, then an exception is thrown.

    Parameters
    ----------
    mol_a: Chem.Mol
        Input molecule a. Must have a conformation.

    mol_b: Chem.Mol
        Input molecule b. Must have a conformation.

    ring_cutoff: float
        The distance cutoff that ring atoms must satisfy.

    chain_cutoff: float
        The distance cutoff that non-ring atoms must satisfy.

    max_visits: int
        Maximum number of nodes we can visit to generate at least one core.

    max_connected_components: int or None
        Set to k to only keep mappings where the number of connected components is <= k.
        The definition of connected here is different from McGregor. Here it means there is a way to reach the mapped
        atom without traversing over a non-mapped atom.

    min_connected_component_size: int
        Set to n to only keep mappings where all connected components have size >= n.

    max_cores: int or float
        maximum number of maximal cores to store, this can be an +np.inf if you want
        every core - when set to 1 this enables a faster predicate that allows for more pruning.

    enforce_core_core: bool
        If we allow core-core bonds to be broken. This may be deprecated later on.

    ring_matches_ring_only: bool
        atom i in mol A can match atom j in mol B
        only if in_ring(i, A) == in_ring(j, B)

    enforce_chiral: bool
        Filter out cores that would flip atom chirality

    disallow_planar_torsion_flips: bool
        Filter out cores that would flip a mapped planar torsion (i.e. change the sign of the torsion volume)

    min_threshold: int
        Number of edges to require for a valid mapping

    initial_mapping: np.ndarray
        Initial core mapping to start the search with

    enforce_chirally_valid_dummy_groups: bool
        Filter out cores that are chirally invalid (i.e. unable to setup chiral restraints at both endstates)

    Returns
    -------
    Returns a list of all_cores

    Raises
    ------
    timemachine.fe.mcgregor.NoMappingError
        If no mapping is found
    """
    all_cores, _ = get_cores_and_diagnostics(
        mol_a,
        mol_b,
        ring_cutoff,
        chain_cutoff,
        max_visits,
        max_connected_components,
        min_connected_component_size,
        max_cores,
        enforce_core_core,
        ring_matches_ring_only,
        enforce_chiral,
        disallow_planar_torsion_flips,
        min_threshold,
        initial_mapping,
        enforce_chirally_valid_dummy_groups,
    )

    return all_cores


def reorder_atoms_by_degree_and_initial_mapping(mol, initial_mapping):
    degrees = [len(a.GetNeighbors()) for a in mol.GetAtoms()]
    for a in mol.GetAtoms():
        if a.GetIdx() in initial_mapping[:, 0]:
            degrees[a.GetIdx()] += np.inf
    perm = np.argsort(degrees, kind="stable")[::-1]

    old_to_new = {}
    for new, old in enumerate(perm):
        old_to_new[old] = new

    new_mol = Chem.RenumberAtoms(mol, perm.tolist())
    new_mapping = []
    for a, b in initial_mapping:
        new_mapping.append([old_to_new[a], b])
    new_mapping = np.array(new_mapping)

    return new_mol, perm, new_mapping


def _uniquify_core(core):
    core_list = []
    for a, b in core:
        core_list.append((a, b))
    return frozenset(core_list)


def _deduplicate_all_cores(all_cores):
    unique_cores = {}
    for core in all_cores:
        # Be careful with the unique core here, list -> set -> list is not consistent
        # across versions of python, use the frozen as the key, but return the untouched
        # cores
        unique_cores[_uniquify_core(core)] = core

    return list(unique_cores.values())


def ring_breaking_count(mol_a, mol_b, core: NDArray) -> Tuple[int, int]:
    """Counts the number of rings broken going forward and reverse given a core. Does not count ring insertion/deletions.

    The algorithm is to add all ring atoms/bonds to a graph, and count the cycles that contain both core and mol a or
    mol b atoms.

    Returns
    -------
    2-tuple
        The rings broken going from mol_a to mol_b and from mol_b to mol_a
    """
    g = nx.Graph()
    map_mixin = AtomMapMixin(mol_a, mol_b, core)
    core_atoms_a = set(core[:, 0])
    core_atoms_b = set(core[:, 1])
    for atom in mol_a.GetAtoms():
        idx = atom.GetIdx()
        if atom.IsInRing():
            if idx in core_atoms_a:
                g.add_node(map_mixin.a_to_c[idx], atom_type=AtomMapFlags.CORE)
            else:
                g.add_node(map_mixin.a_to_c[idx], atom_type=AtomMapFlags.MOL_A)
    for atom in mol_b.GetAtoms():
        idx = atom.GetIdx()
        if atom.IsInRing() and idx not in core_atoms_b:
            g.add_node(map_mixin.b_to_c[idx], atom_type=AtomMapFlags.MOL_B)

    # if there are no ring atoms, return 0
    if len(g.nodes) == 0:
        return (0, 0)

    for bond in mol_a.GetBonds():
        if bond.IsInRing():
            src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if src in g.nodes and dst in g.nodes:
                g.add_edge(src, dst)

    for bond in mol_b.GetBonds():
        if bond.IsInRing():
            src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src = map_mixin.b_to_c[src]
            dst = map_mixin.b_to_c[dst]
            if src in g.nodes and dst in g.nodes:
                g.add_edge(src, dst)

    # Need at least three edges to make a cycle
    if len(g.edges) < 3:
        return (0, 0)

    def core_dummy_bond(graph: nx.Graph, e: Tuple[int, int]) -> bool:
        """Determine if a bond is between two nodes of the same type. If not it is a core-dummy bond. Expects that one
        of the atom types is AtomMapFlags.CORE and the other is AtomMapFlags.MOL_A or AtomMapFlags.MOL_B
        """
        a, b = e
        return graph.nodes[a]["atom_type"] != graph.nodes[b]["atom_type"]

    breaking_count = [0, 0]
    for i, comp in enumerate(nx.connected_components(g)):
        subgraph = g.subgraph(comp)
        # If the nodes are all core atoms, not a ring breaking transformation.
        if all([data["atom_type"] == AtomMapFlags.CORE for _, data in subgraph.nodes(data=True)]):
            continue

        for endstate_idx, atom_type in enumerate((AtomMapFlags.MOL_A, AtomMapFlags.MOL_B)):
            ring_atoms = subgraph.subgraph(
                [n for n, data in subgraph.nodes(data=True) if data["atom_type"] in (AtomMapFlags.CORE, atom_type)]
            )
            core_ring_atom_count = np.sum(
                [data["atom_type"] == AtomMapFlags.CORE for _, data in ring_atoms.nodes(data=True)], dtype=np.int32
            )
            # If is only one or fewer core atoms, it is not a ring forming/breaking transformation and just an insertion
            if core_ring_atom_count <= 1:
                continue
            # If the ring atoms are all core atoms, no ring break has occurred
            elif core_ring_atom_count == len(ring_atoms):
                continue
            try:
                # Make a copy of the ring_atoms graph so that it can be modified
                ring_atoms = nx.Graph(ring_atoms)
                while len(ring_atoms.edges) >= 3:
                    cycle_edges = nx.find_cycle(ring_atoms)
                    core_dummy_bond_mask = [core_dummy_bond(ring_atoms, e) for e in cycle_edges]
                    # Only interested in cycles constructed with a core-dummy bond
                    if any(core_dummy_bond_mask):
                        breaking_count[endstate_idx] += 1
                        edge_to_remove = core_dummy_bond_mask.index(True)
                    else:
                        # Pick an arbitrary edge to delete to break the cycle
                        edge_to_remove = 0
                    # Delete an edge to break the existing cycle
                    ring_atoms.remove_edge(*cycle_edges[edge_to_remove])
            except nx.exception.NetworkXNoCycle:
                pass
    return breaking_count[0], breaking_count[1]


def core_bonds_broken_count(mol_a, mol_b, core):
    # count the number of core bonds broken in mol_a when applying the core atom map
    core_a_to_b = dict(core)
    count = 0
    for bond in mol_a.GetBonds():
        src_a, dst_a = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if src_a in core_a_to_b and dst_a in core_a_to_b:
            if mol_b.GetBondBetweenAtoms(int(core_a_to_b[src_a]), int(core_a_to_b[dst_a])):
                pass
            else:
                count += 1

    return count


def _get_cores_impl(
    mol_a,
    mol_b,
    ring_cutoff,
    chain_cutoff,
    max_visits,
    max_connected_components: Optional[int],
    min_connected_component_size: int,
    max_cores,
    enforce_core_core,
    ring_matches_ring_only,
    enforce_chiral,
    disallow_planar_torsion_flips,
    min_threshold,
    initial_mapping,
    enforce_chirally_valid_dummy_groups: bool,
) -> Tuple[List[NDArray], mcgregor.MCSDiagnostics]:
    if initial_mapping is None:
        initial_mapping = np.zeros((0, 2))

    mol_a, perm, initial_mapping = reorder_atoms_by_degree_and_initial_mapping(mol_a, initial_mapping)

    bonds_a = get_romol_bonds(mol_a)
    bonds_b = get_romol_bonds(mol_b)
    conf_a = get_romol_conf(mol_a)
    conf_b = get_romol_conf(mol_b)

    priority_idxs = []  # ordered list of atoms to consider

    # setup co-domain for each atom in mol_a, if an initial mapping is provided, it overrides
    # the priority_idxs
    initial_mapping_kv = {}
    for src, dst in initial_mapping:
        initial_mapping_kv[src] = dst

    for idx, a_xyz in enumerate(conf_a):
        if idx < len(initial_mapping):
            priority_idxs.append([initial_mapping_kv[idx]])  # used to initialize marcs and nothing else
        else:
            atom_i = mol_a.GetAtomWithIdx(idx)
            dijs = []

            allowed_idxs = set()
            for jdx, b_xyz in enumerate(conf_b):
                atom_j = mol_b.GetAtomWithIdx(jdx)
                dij = np.linalg.norm(a_xyz - b_xyz)
                dijs.append(dij)

                if ring_matches_ring_only and (atom_i.IsInRing() != atom_j.IsInRing()):
                    continue

                cutoff = ring_cutoff if (atom_i.IsInRing() or atom_j.IsInRing()) else chain_cutoff
                if dij < cutoff:
                    allowed_idxs.add(jdx)

            final_idxs = []
            for idx in np.argsort(dijs, kind="stable"):
                if idx in allowed_idxs:
                    final_idxs.append(idx)

            priority_idxs.append(final_idxs)

    n_a = len(conf_a)
    n_b = len(conf_b)

    filter_fxns = []
    if enforce_chiral:
        chiral_set_a = ChiralRestrIdxSet.from_mol(mol_a, conf_a)
        chiral_set_b = ChiralRestrIdxSet.from_mol(mol_b, conf_b)

        def chiral_filter(trial_core):
            passed = not has_chiral_atom_flips(trial_core, chiral_set_a, chiral_set_b)
            return passed

        filter_fxns.append(chiral_filter)

    if disallow_planar_torsion_flips:
        find_flipped_planar_torsions = setup_find_flipped_planar_torsions(mol_a, mol_b)

        def planar_torsion_flip_filter(trial_core):
            flipped = find_flipped_planar_torsions(trial_core)
            passed = next(flipped, None) is None  # i.e. iterator is empty
            return passed

        filter_fxns.append(planar_torsion_flip_filter)

    def filter_fxn(trial_core):
        return all(f(trial_core) for f in filter_fxns)

    def make_leaf_filter_fxn():
        find_chirally_valid_dummy_groups = make_find_chirally_valid_dummy_groups(mol_a, mol_b)

        def leaf_filter_fxn(trial_core) -> bool:
            if enforce_chirally_valid_dummy_groups:
                core = mcgregor.perm_to_core(trial_core)
                chirally_valid_dummy_groups = find_chirally_valid_dummy_groups(core)
                return chirally_valid_dummy_groups is not None
            else:
                return True

        return leaf_filter_fxn

    all_cores, _, mcs_diagnostics = mcgregor.mcs(
        n_a,
        n_b,
        priority_idxs,
        bonds_a,
        bonds_b,
        max_visits,
        max_cores,
        enforce_core_core,
        max_connected_components,
        min_connected_component_size,
        min_threshold,
        initial_mapping,
        filter_fxn,
        make_leaf_filter_fxn(),
    )

    all_cores = remove_cores_smaller_than_largest(all_cores)
    all_cores = _deduplicate_all_cores(all_cores)

    mean_sq_distances = []
    valence_mismatches = []
    cb_counts = []

    for core in all_cores:
        r_i = conf_a[core[:, 0]]
        r_j = conf_b[core[:, 1]]

        # distance score
        r2_ij = np.sum(np.power(r_i - r_j, 2))
        # No need to perform square root here, only care about ordering
        mean_sq_dist = r2_ij / len(core)
        mean_sq_distances.append(mean_sq_dist)

        v_count = 0
        for idx, jdx in core:
            v_count += abs(
                mol_a.GetAtomWithIdx(int(idx)).GetTotalValence() - mol_b.GetAtomWithIdx(int(jdx)).GetTotalValence()
            )

        valence_mismatches.append(v_count)
        cb_counts.append(
            core_bonds_broken_count(mol_a, mol_b, core) + core_bonds_broken_count(mol_b, mol_a, core[:, [1, 0]])
        )

    sort_vals = np.array(
        list(zip(cb_counts, valence_mismatches, mean_sq_distances)), dtype=[("cb", "i"), ("valence", "i"), ("msd", "f")]
    )
    sorted_cores = []

    sort_order = np.argsort(sort_vals, order=["cb", "valence", "msd"])
    for p in sort_order:
        core = all_cores[p]
        # undo the sort
        core[:, 0] = perm[core[:, 0]]
        sorted_cores.append(core)

    return sorted_cores, mcs_diagnostics


def remove_cores_smaller_than_largest(cores):
    """measured by # mapped atoms"""
    cores_by_size = defaultdict(list)
    for core in cores:
        cores_by_size[len(core)].append(core)

    # Return the largest core(s)
    max_core_size = max(cores_by_size.keys())
    return cores_by_size[max_core_size]
