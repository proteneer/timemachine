from collections import defaultdict
from typing import List, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

from timemachine.fe import mcgregor
from timemachine.fe.chiral_utils import ChiralRestrIdxSet, has_chiral_atom_flips, setup_find_flipped_planar_torsions
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
    connected_core,
    max_cores,
    enforce_core_core,
    ring_matches_ring_only,
    complete_rings,
    enforce_chiral,
    disallow_planar_torsion_flips,
    min_threshold,
) -> Tuple[List[NDArray], mcgregor.MCSDiagnostics]:
    """Same as :py:func:`get_cores`, but additionally returns diagnostics collected during the MCS search."""
    assert max_cores > 0

    core_kwargs = dict(
        ring_cutoff=ring_cutoff,
        chain_cutoff=chain_cutoff,
        max_visits=max_visits,
        connected_core=connected_core,
        max_cores=max_cores,
        enforce_core_core=enforce_core_core,
        ring_matches_ring_only=ring_matches_ring_only,
        complete_rings=complete_rings,
        enforce_chiral=enforce_chiral,
        disallow_planar_torsion_flips=disallow_planar_torsion_flips,
        min_threshold=min_threshold,
    )

    # we require that mol_a.GetNumAtoms() <= mol_b.GetNumAtoms()
    if mol_a.GetNumAtoms() > mol_b.GetNumAtoms():
        all_cores, mcs_diagnostics = _get_cores_impl(mol_b, mol_a, **core_kwargs)
        new_cores = []
        for core in all_cores:
            core = np.array([(x[1], x[0]) for x in core], dtype=core.dtype)
            new_cores.append(core)
        return new_cores, mcs_diagnostics
    else:
        all_cores, mcs_diagnostics = _get_cores_impl(mol_a, mol_b, **core_kwargs)
        return all_cores, mcs_diagnostics


def get_cores(
    mol_a,
    mol_b,
    ring_cutoff,
    chain_cutoff,
    max_visits,
    connected_core,
    max_cores,
    enforce_core_core,
    ring_matches_ring_only,
    complete_rings,
    enforce_chiral,
    disallow_planar_torsion_flips,
    min_threshold,
) -> List[NDArray]:
    """
    Finds set of cores between two molecules that maximizes the number of common edges.

    If either atom i or atom j is in a ring then the dist(i,j) < ring_cutoff, otherwise dist(i,j) < chain_cutoff

    Additional notes
    ----------------
    1) The returned cores are sorted in increasing order based on the rmsd of the alignment.
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
        Maximum number of nodes we can visit for a given threshold.

    connected_core: bool
        Set to True to only keep the largest connected
        subgraph in the mapping. The definition of connected
        here is different from McGregor. Here it means there
        is a way to reach the mapped atom without traversing
        over a non-mapped atom.

    max_cores: int or float
        maximum number of maximal cores to store, this can be an +np.inf if you want
        every core - when set to 1 this enables a faster predicate that allows for more pruning.

    enforce_core_core: bool
        If we allow core-core bonds to be broken. This may be deprecated later on.

    ring_matches_ring_only: bool
        atom i in mol A can match atom j in mol B
        only if in_ring(i, A) == in_ring(j, B)

    complete_rings: bool
        If we require mapped atoms that are in a ring to be complete.
        If True then connected_core must also be True.

    enforce_chiral: bool
        Filter out cores that would flip atom chirality

    disallow_planar_torsion_flips: bool
        Filter out cores that would flip a mapped planar torsion (i.e. change the sign of the torsion volume)

    min_threshold: int
        Number of atoms to require for a valid mapping

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
        connected_core,
        max_cores,
        enforce_core_core,
        ring_matches_ring_only,
        complete_rings,
        enforce_chiral,
        disallow_planar_torsion_flips,
        min_threshold,
    )

    return all_cores


def bfs(g, atom):
    depth = 0
    cur_layer = [atom]
    levels = {}
    while len(levels) != g.GetNumAtoms():
        next_layer = []
        for layer_atom in cur_layer:
            levels[layer_atom.GetIdx()] = depth
            for nb_atom in layer_atom.GetNeighbors():
                if nb_atom.GetIdx() not in levels:
                    next_layer.append(nb_atom)
        cur_layer = next_layer
        depth += 1
    levels_array = [-1] * g.GetNumAtoms()
    for i, l in levels.items():
        levels_array[i] = l
    return levels_array


def reorder_atoms_by_degree(mol):
    degrees = [len(a.GetNeighbors()) for a in mol.GetAtoms()]
    perm = np.argsort(degrees, kind="stable")[::-1]
    new_mol = Chem.RenumberAtoms(mol, perm.tolist())
    return new_mol, perm


def find_cycles(g: nx.Graph):
    # return the indices of nxg that are in a cycle
    # 1) find and remove bridges
    # 2) if atom has > 1 neighbor then it is in a cycle.
    edges = nx.bridges(g)
    for e in edges:
        g.remove_edge(*e)
    cycle_dict = {}
    for node in g.nodes:
        # print(list(g[node]))
        cycle_dict[node] = len(list(g.neighbors(node))) > 0
    return cycle_dict


def get_edges(mol):
    return [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]


def induce_mol_subgraph(mol_a, core_a, bond_core_a):
    sg_a = nx.Graph()
    for a in core_a:
        sg_a.add_node(a)
    for e1 in bond_core_a:
        sg_a.add_edge(*e1)
    return sg_a


def _to_networkx_graph(mol):
    g = nx.Graph()
    for atom in mol.GetAtoms():
        g.add_node(atom.GetIdx())

    for bond in mol.GetBonds():
        src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        g.add_edge(src, dst)
    return g


def _remove_incomplete_rings(mol_a, mol_b, core, bond_core):
    # to networkx
    g_a, g_b = _to_networkx_graph(mol_a), _to_networkx_graph(mol_b)
    sg_a = induce_mol_subgraph(mol_a, core[:, 0], list(bond_core.keys()))
    sg_b = induce_mol_subgraph(mol_b, core[:, 1], list(bond_core.values()))

    a_cycles = find_cycles(g_a)
    b_cycles = find_cycles(g_b)
    sg_a_cycles = find_cycles(sg_a)
    sg_b_cycles = find_cycles(sg_b)

    new_core = []

    for a, b in core:
        pred_sgg_a = a_cycles[a] == sg_a_cycles[a]
        pred_sgg_b = b_cycles[b] == sg_b_cycles[b]
        # a and b are consistent
        pred_sg_ab = sg_a_cycles[a] == sg_b_cycles[b]
        pred_g_ab = a_cycles[a] == b_cycles[b]
        # all four have to be consistent
        if pred_sgg_a and pred_sgg_b and pred_sg_ab and pred_g_ab:
            new_core.append([a, b])

    final_core = np.array(new_core)
    final_bond_core = update_bond_core(final_core, bond_core)
    return final_core, final_bond_core


def remove_incomplete_rings(mol_a, mol_b, all_cores, all_bond_cores):
    new_cores = []
    new_bond_cores = []
    for cores, bond_cores in zip(all_cores, all_bond_cores):
        nc, nbc = _remove_incomplete_rings(mol_a, mol_b, cores, bond_cores)
        new_cores.append(nc)
        new_bond_cores.append(nbc)
    return new_cores, new_bond_cores


def _compute_bond_cores(mol_a, mol_b, marcs):
    a_edges = get_edges(mol_a)
    b_edges = get_edges(mol_b)
    bond_core = {}
    for e_a in range(len(a_edges)):
        src_a, dst_a = a_edges[e_a]
        for e_b in range(len(b_edges)):
            src_b, dst_b = b_edges[e_b]
            if marcs[e_a][e_b]:
                assert (src_a, dst_a) not in bond_core
                assert (dst_a, src_a) not in bond_core
                bond_core[(src_a, dst_a)] = (src_b, dst_b)
    return bond_core


def _uniquify_core(core):
    core_list = []
    for a, b in core:
        core_list.append((a, b))
    return frozenset(core_list)


def _deduplicate_all_cores_and_bonds(all_cores, all_bonds):
    unique_cores = {}
    for core, bond in zip(all_cores, all_bonds):
        # Be careful with the unique core here, list -> set -> list is not consistent
        # across versions of python, use the frozen as the key, but return the untouched
        # cores
        unique_cores[_uniquify_core(core)] = (core, bond)

    cores = []
    bonds = []
    for core, bond_core in unique_cores.values():
        cores.append(np.array(core))
        bonds.append(bond_core)
    return cores, bonds


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
    connected_core,
    max_cores,
    enforce_core_core,
    ring_matches_ring_only,
    complete_rings,
    enforce_chiral,
    disallow_planar_torsion_flips,
    min_threshold,
) -> Tuple[List[NDArray], mcgregor.MCSDiagnostics]:
    mol_a, perm = reorder_atoms_by_degree(mol_a)  # UNINVERT

    bonds_a = get_romol_bonds(mol_a)
    bonds_b = get_romol_bonds(mol_b)
    conf_a = get_romol_conf(mol_a)
    conf_b = get_romol_conf(mol_b)

    priority_idxs = []  # ordered list of atoms to consider

    # setup co-domain for each atom in mol_a
    for idx, a_xyz in enumerate(conf_a):
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

    all_cores, all_marcs, mcs_diagnostics = mcgregor.mcs(
        n_a,
        n_b,
        priority_idxs,
        bonds_a,
        bonds_b,
        max_visits,
        max_cores,
        enforce_core_core,
        min_threshold,
        filter_fxn=filter_fxn,
    )

    all_bond_cores = [_compute_bond_cores(mol_a, mol_b, marcs) for marcs in all_marcs]

    if connected_core and complete_rings:
        # 1) remove any disconnected components or lone atoms in the mapping
        # so disconnected regions are not considered as part of the incomplete rings check.
        # 2) deduplicate cores, to avoid checking incomplete rings on the same core
        # 3) remove any incomplete rings
        # 4) deduplicate cores, to avoid removing disconnections on the same core
        # 5) remove any disconnections created by removing the incomplete rings.
        all_cores, all_bond_cores = remove_disconnected_components(mol_a, mol_b, all_cores, all_bond_cores)
        all_cores, all_bond_cores = _deduplicate_all_cores_and_bonds(all_cores, all_bond_cores)

        all_cores, all_bond_cores = remove_incomplete_rings(mol_a, mol_b, all_cores, all_bond_cores)
        all_cores, all_bond_cores = _deduplicate_all_cores_and_bonds(all_cores, all_bond_cores)

        all_cores, all_bond_cores = remove_disconnected_components(mol_a, mol_b, all_cores, all_bond_cores)
    elif connected_core and not complete_rings:
        all_cores, all_bond_cores = remove_disconnected_components(mol_a, mol_b, all_cores, all_bond_cores)
    elif not connected_core and complete_rings:
        all_cores, all_bond_cores = remove_incomplete_rings(mol_a, mol_b, all_cores, all_bond_cores)

    all_cores = remove_cores_smaller_than_largest(all_cores)
    all_cores, _ = _deduplicate_all_cores_and_bonds(all_cores, all_bond_cores)

    dists = []
    valence_mismatches = []
    cb_counts = []

    for core in all_cores:
        r_i = conf_a[core[:, 0]]
        r_j = conf_b[core[:, 1]]

        # distance score
        r2_ij = np.sum(np.power(r_i - r_j, 2))
        rmsd = np.sqrt(r2_ij / len(core))
        dists.append(rmsd)

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
        list(zip(cb_counts, valence_mismatches, dists)), dtype=[("cb", "i"), ("valence", "i"), ("rmsd", "f")]
    )
    sorted_cores = []

    sort_order = np.argsort(sort_vals, order=["cb", "valence", "rmsd"])
    for p in sort_order:
        sorted_cores.append(all_cores[p])

    # undo the sort
    for core in sorted_cores:
        inv_core = []
        for atom in core[:, 0]:
            inv_core.append(perm[atom])
        core[:, 0] = inv_core

    return sorted_cores, mcs_diagnostics


# maintainer: jkaus
def remove_disconnected_components(mol_a, mol_b, all_cores, all_bond_cores):
    """
    This will remove all but the largest connected component from each core map.
    Even if two adjacent atoms are both mapped, their bond may not be present in
    bond_cores, indicating a disconnection. This will iteratively remove the smaller
    components from each mapping, until both molecules have only a single
    connected component in their maps.
    """
    filtered_cores = []
    filtered_bond_cores = []
    for core, bond_core in zip(all_cores, all_bond_cores):
        new_core = core
        new_bond_core = bond_core
        # Need to run it once through even if fully connected
        # to remove stray atoms that are not included in the bond_core
        first = True
        while True:
            core_a = list(new_core[:, 0])
            core_b = list(new_core[:, 1])

            g_mol_a = nx.Graph()
            g_mol_b = nx.Graph()
            for bond_a, bond_b in new_bond_core.items():
                g_mol_a.add_edge(*bond_a)
                g_mol_b.add_edge(*bond_b)

            cc_a = list(nx.connected_components(g_mol_a))
            cc_b = list(nx.connected_components(g_mol_b))

            # stop when the core is fully connected
            if not first and len(cc_a) == 1 and len(cc_b) == 1:
                break
            # No atoms left to map
            elif len(cc_a) == 0 or len(cc_b) == 0:
                new_core = []
                break

            largest_cc_a = max(cc_a, key=len)
            largest_cc_b = max(cc_b, key=len)

            new_core_idxs = []
            if len(largest_cc_a) < len(largest_cc_b):
                # mol_a has the smaller cc
                for atom_idx in largest_cc_a:
                    core_idx = core_a.index(atom_idx)
                    new_core_idxs.append(core_idx)
            else:
                # mol_b has the smaller cc
                for atom_idx in largest_cc_b:
                    core_idx = core_b.index(atom_idx)
                    new_core_idxs.append(core_idx)

            new_core = new_core[new_core_idxs]
            new_bond_core = update_bond_core(new_core, new_bond_core)
            first = False

        if len(new_core) == 0:
            continue
        filtered_cores.append(new_core)
        filtered_bond_cores.append(new_bond_core)

    return filtered_cores, filtered_bond_cores


def remove_cores_smaller_than_largest(cores):
    """measured by # mapped atoms"""
    cores_by_size = defaultdict(list)
    for core in cores:
        cores_by_size[len(core)].append(core)

    # Return the largest core(s)
    max_core_size = max(cores_by_size.keys())
    return cores_by_size[max_core_size]


def update_bond_core(core, bond_core):
    """
    bond_core: dictionary mapping atoms (i,j) of mol a
    to (k, l) of mol b.
    """
    new_bond_core = {}
    core_a = list(core[:, 0])
    core_b = list(core[:, 1])
    for bond_a, bond_b in bond_core.items():
        if bond_a[0] in core_a and bond_a[1] in core_a and bond_b[0] in core_b and bond_b[1] in core_b:
            new_bond_core[bond_a] = bond_b
    return new_bond_core
