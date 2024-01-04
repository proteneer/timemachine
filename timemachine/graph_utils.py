import networkx as nx


def convert_to_nx(mol):
    """
    Convert an ROMol into a networkx graph.
    """
    g = nx.Graph()
    for atom in mol.GetAtoms():
        g.add_node(atom.GetIdx())

    for bond in mol.GetBonds():
        src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        g.add_edge(src, dst)

    return g


def enumerate_simple_paths_from(graph, start_node, cutoff):
    def go(node, cutoff, visited):
        if cutoff == 1:
            return [[node]]
        return [
            [node] + path
            for neighbor in nx.neighbors(graph, node)
            if neighbor not in visited
            for path in go(neighbor, cutoff - 1, visited | {node})
        ]

    return go(start_node, cutoff, set())


def enumerate_simple_paths(graph, cutoff):
    return [path for start_node in graph for path in enumerate_simple_paths_from(graph, start_node, cutoff)]
