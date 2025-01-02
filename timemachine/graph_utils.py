from typing import List, TypeVar

import networkx as nx


def convert_to_nx(mol):
    """
    Convert an Chem.Mol into a networkx graph.
    """
    g = nx.Graph()
    for atom in mol.GetAtoms():
        g.add_node(atom.GetIdx())

    for bond in mol.GetBonds():
        src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        g.add_edge(src, dst)

    return g


_Node = TypeVar("_Node")


def enumerate_simple_paths_from(graph: nx.Graph, start_node: _Node, length: int) -> List[List[_Node]]:
    """Return all simple paths of a given length starting from a given node.

    A simple path is a path without repeated nodes.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph

    start_node : node
        Initial node for all paths

    length : int
        Length of returned paths

    Returns
    -------
    list of list of node
        Simple paths
    """

    def go(node, cutoff, visited):
        if cutoff == 1:
            return [[node]]
        return [
            [node] + path
            for neighbor in nx.neighbors(graph, node)
            if neighbor not in visited
            for path in go(neighbor, cutoff - 1, visited | {node})
        ]

    return go(start_node, length, set())


def enumerate_simple_paths(graph: nx.Graph, length: int) -> List[List]:
    """Return all simple paths of a given length.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph

    length : int
        Length of returned paths

    Returns
    -------
    list of list of node
        Simple paths
    """
    return [path for start_node in graph for path in enumerate_simple_paths_from(graph, start_node, length)]
