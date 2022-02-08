from typing import List

import networkx as nx
import numpy as np
import simtk.unit


def to_md_units(q):
    return q.value_in_unit_system(simtk.unit.md_unit_system)


def write(xyz, masses, recenter=True):
    if recenter:
        xyz = xyz - np.mean(xyz, axis=0, keepdims=True)
    buf = str(len(masses)) + "\n"
    buf += "timemachine\n"
    for m, (x, y, z) in zip(masses, xyz):
        if int(round(m)) == 12:
            symbol = "C"
        elif int(round(m)) == 14:
            symbol = "N"
        elif int(round(m)) == 16:
            symbol = "O"
        elif int(round(m)) == 32:
            symbol = "S"
        elif int(round(m)) == 35:
            symbol = "Cl"
        elif int(round(m)) == 1:
            symbol = "H"
        elif int(round(m)) == 31:
            symbol = "P"
        elif int(round(m)) == 19:
            symbol = "F"
        elif int(round(m)) == 80:
            symbol = "Br"
        elif int(round(m)) == 127:
            symbol = "I"
        else:
            raise Exception("Unknown mass:" + str(m))

        buf += symbol + " " + str(round(x, 5)) + " " + str(round(y, 5)) + " " + str(round(z, 5)) + "\n"
    return buf


def convert_uIC50_to_kJ_per_mole(amount_in_uM):
    """
    TODO: more sig figs
    """
    return 0.593 * np.log(amount_in_uM * 1e-6) * 4.18


def convert_uM_to_kJ_per_mole(amount_in_uM):
    """
    Convert a potency measurement in uM concentrations.

    Parameters
    ----------
    amount_in_uM: float
        Binding potency in uM concentration.

    Returns
    -------
    float
        Binding potency in kJ/mol.

    """
    return 0.593 * np.log(amount_in_uM * 1e-6) * 4.18


from scipy.spatial.distance import cdist


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


# TODO: add a module for atom-mapping, with RDKit MCS based and other approaches

# TODO: add a visualization module?
# TODO: compare with perses atom map visualizations?

from rdkit.Chem.Draw import rdMolDraw2D


def draw_mol(mol, highlightAtoms, highlightColors):
    """from YTZ, Feb 1, 2021"""
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 200)
    drawer.DrawMolecule(mol, highlightAtoms=highlightAtoms, highlightAtomColors=highlightColors)
    drawer.FinishDrawing()

    # TODO: return or save image, for inclusion in a PDF report or similar

    # To display in a notebook:
    # svg = drawer.GetDrawingText().replace('svg:', '')
    # display(SVG(svg))


def plot_atom_mapping(mol_a, mol_b, core):
    """from YTZ, Feb 1, 2021

    TODO: move this into a SingleTopology.visualize() or SingleTopology.debug() method"""
    print(repr(core))
    atom_colors_a = {}
    atom_colors_b = {}
    for (a_idx, b_idx), rgb in zip(core, np.random.random((len(core), 3))):
        atom_colors_a[int(a_idx)] = tuple(rgb.tolist())
        atom_colors_b[int(b_idx)] = tuple(rgb.tolist())

    draw_mol(mol_a, core[:, 0].tolist(), atom_colors_a)
    draw_mol(mol_b, core[:, 1].tolist(), atom_colors_b)


def get_connected_components(nodes, relative_inds, absolute_inds) -> List[np.array]:
    """Construct a graph containing (len(nodes) + 1) nodes -- one for each original node, plus a new "reference" node.*

    Add edges
    * (i, j) in relative_inds,
    * (i, "reference") for i in absolute_inds

    And then return the connected components of this graph (omitting the "reference" node we added).*

    * Unless "nodes" already contained something named "reference"!
    """
    g = nx.Graph()
    g.add_nodes_from(nodes)

    if "reference" not in nodes:
        g.add_node("reference")

    for (i, j) in relative_inds:
        g.add_edge(i, j)

    if len(absolute_inds) == 0:
        absolute_inds = [0]

    for i in absolute_inds:
        g.add_edge(i, "reference")

    # return list of lists of elements of the nodes
    # we will remove the "reference" node we added
    # however, if the user actually had a node named "reference", don't remove it

    components = list(map(list, list(nx.connected_components(g))))

    if "reference" in nodes:
        return components
    else:
        filtered_components = []

        for component in components:
            if "reference" in component:
                component.remove("reference")
            filtered_components.append(component)
        return filtered_components


def validate_map(n_nodes: int, relative_inds: np.array, absolute_inds: np.array) -> bool:
    """Construct a graph containing (n_nodes + 1) nodes -- one for each original node, plus a new "reference" node.

    Add edges
    * (i, j) in relative_inds,
    * (i, "reference") for i in absolute_inds

    And then return whether this graph is connected.

    If no absolute_inds provided, treat node 0 as "reference".

    Examples
    --------

    >>> validate_map(4, relative_inds=[[0,1], [2,3]], absolute_inds=[0])
    False

    >>> validate_map(4, relative_inds=[[0,1], [1,2], [2,3]], absolute_inds=[0])
    True

    >>> validate_map(4, relative_inds=[[0,1], [2,3]], absolute_inds=[0,2])
    True
    """

    if len(absolute_inds) == 0:
        absolute_inds = [0]

    components = get_connected_components(list(range(n_nodes)), relative_inds, absolute_inds)
    return len(components) == 1


def get_romol_conf(mol):
    """Coordinates of mol's 0th conformer, in nanometers"""
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf / 10  # from angstroms to nm


def sanitize_energies(full_us, lamb_idx, cutoff=10000):
    """
    Given a matrix with F rows and K columns,
    we sanitize entries that differ by more than cutoff.

    That is, given full_us:
    [
        [15000.0, -5081923.0, 1598, 1.5, -23.0],
        [-423581.0, np.nan, -238, 13.5,  23.0]
    ]
    And lamb_idx 3 and cutoff of 10000,
    full_us is sanitized to:

    [
        [inf, inf, 1598, 1.5, -23.0],
        [inf, inf, -238, 13.5,  23.0]
    ]

    Parameters
    ----------
    full_us: np.array of shape (F, K)
        Matrix of full energies

    lamb_idx: int
        Which of the K windows to serve as the reference energy

    cutoff: float
        Used to determine the threshold for a "good" energy

    Returns
    -------
    np.array of shape (F,K)
        Sanitized energies

    """
    ref_us = np.expand_dims(full_us[:, lamb_idx], axis=1)
    abs_us = np.abs(full_us - ref_us)
    return np.where(abs_us < cutoff, full_us, np.inf)


def extract_delta_Us_from_U_knk(U_knk):
    """
    Generate delta_Us from the U_knk matrix for use with BAR.

    Parameters
    ----------
    U_knk: np.array of shape (K, N, K)
        Energies matrix, K simulations ran with N frames with
        energies evaluated at K states

    Returns
    -------
    np.array of shape (K-1, 2, N)
        Returns the delta_Us of the fwd and rev processes

    """

    assert U_knk.shape[0] == U_knk.shape[-1]

    K = U_knk.shape[0]

    def delta_U(from_idx, to_idx):
        """
        Computes [U(x, to_idx) - U(x, from_idx) for x in xs]
        where xs are simulated at from_idx
        """
        current = U_knk[from_idx]
        current_energies = current[:, from_idx]
        perturbed_energies = current[:, to_idx]
        return perturbed_energies - current_energies

    delta_Us = []

    for lambda_idx in range(K - 1):
        # lambda_us have shape (F, K)
        fwd_delta_U = delta_U(lambda_idx, lambda_idx + 1)
        rev_delta_U = delta_U(lambda_idx + 1, lambda_idx)
        delta_Us.append((fwd_delta_U, rev_delta_U))

    return np.array(delta_Us)
