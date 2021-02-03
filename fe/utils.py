import numpy as np
import simtk.unit

def set_velocities_to_temperature(n_atoms, temperature, masses):
    assert 0 # don't call this yet until its
    v_t = np.random.normal(size=(n_atoms, 3))
    velocity_scale = np.sqrt(constants.BOLTZ*temperature/np.expand_dims(masses, -1))
    return v_t*velocity_scale

def to_md_units(q):
    return q.value_in_unit_system(simtk.unit.md_unit_system)

def write(xyz, masses, recenter=True):
    if recenter:
        xyz = xyz - np.mean(xyz, axis=0, keepdims=True)
    buf = str(len(masses)) + '\n'
    buf += 'timemachine\n'
    for m, (x,y,z) in zip(masses, xyz):
        if int(round(m)) == 12:
            symbol = 'C'
        elif int(round(m)) == 14:
            symbol = 'N'
        elif int(round(m)) == 16:
            symbol = 'O'
        elif int(round(m)) == 32:
            symbol = 'S'
        elif int(round(m)) == 35:
            symbol = 'Cl'
        elif int(round(m)) == 1:
            symbol = 'H'
        elif int(round(m)) == 31:
            symbol = 'P'
        elif int(round(m)) == 19:
            symbol = 'F'
        elif int(round(m)) == 80:
            symbol = 'Br'
        elif int(round(m)) == 127:
            symbol = 'I'
        else:
            raise Exception("Unknown mass:" + str(m))

        buf += symbol + ' ' + str(round(x,5)) + ' ' + str(round(y,5)) + ' ' +str(round(z,5)) + '\n'
    return buf


def convert_uIC50_to_kJ_per_mole(amount_in_uM):
    """
    TODO: more sig figs
    """
    return 0.593 * np.log(amount_in_uM * 1e-6) * 4.18


from scipy.spatial.distance import cdist
import networkx as nx

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
            g.add_edge(f'conf_a[{i}]', f'conf_b[{j}]', weight=threshold - distances[i, j])
    return g


def _core_from_matching(matching):
    """matching is a set of pairs of node names"""

    # 'conf_b[9]' -> 9
    ind_from_node_name = lambda name: int(name.split('[')[1].split(']')[0])

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

# TODO: add a module for atom-mapping, with RDKit MCS based and other approaches