from enum import IntEnum

import networkx as nx

from timemachine.fe.dummy import canonicalize_bond


class NodeState(IntEnum):
    OFF = 0
    ON = 1


class NodeType(IntEnum):
    BOND = 0
    ANGLE = 1
    PROPER_TORSION = 2
    IMPROPER_TORSION = 3
    CHIRAL_ATOM = 4


def _get_implied_bond_nodes_from_angle_idxs(i, j, k):
    implied_bond_idxs = [canonicalize_bond([i, j]), canonicalize_bond([j, k])]
    bond_nodes = [(NodeType.BOND, tuple(bond_idxs)) for bond_idxs in implied_bond_idxs]
    return bond_nodes


def _get_implied_bond_nodes_from_proper_idxs(i, j, k, l):
    implied_bond_idxs = [canonicalize_bond([i, j]), canonicalize_bond([j, k]), canonicalize_bond([k, l])]
    bond_nodes = [(NodeType.BOND, tuple(bond_idxs)) for bond_idxs in implied_bond_idxs]
    return bond_nodes


def _get_implied_angle_nodes_from_proper_idxs(i, j, k, l):
    implied_angle_idxs = [
        canonicalize_bond([i, j, k]),
        canonicalize_bond([j, k, l]),
    ]
    angle_nodes = [(NodeType.ANGLE, tuple(angle_idxs)) for angle_idxs in implied_angle_idxs]
    return angle_nodes


def _get_implied_bond_nodes_from_improper_idxs(c, j, k, l):
    implied_bond_idxs = [canonicalize_bond([c, j]), canonicalize_bond([c, k]), canonicalize_bond([c, l])]
    bond_nodes = [(NodeType.BOND, tuple(bond_idxs)) for bond_idxs in implied_bond_idxs]
    return bond_nodes


def _get_implied_angle_nodes_from_improper_idxs(c, j, k, l):
    implied_angle_idxs = [
        canonicalize_bond([j, c, k]),
        canonicalize_bond([j, c, l]),
        canonicalize_bond([k, c, l]),
    ]
    angle_nodes = [(NodeType.ANGLE, tuple(angle_idxs)) for angle_idxs in implied_angle_idxs]
    return angle_nodes


_get_implied_bond_nodes_from_chiral_atoms = _get_implied_bond_nodes_from_improper_idxs
_get_implied_angle_nodes_from_chiral_atoms = _get_implied_angle_nodes_from_improper_idxs


class DepGraph:
    def __init__(self, bond, angle, proper_torsion, improper_torsion, chiral_atom):
        # self.potentials = potentials
        self._dag = nx.DiGraph()

        # primitives
        # 1. add bonds
        # 3. add angles (depends on bonds and chiral volumes)
        # 2. add chiral volumes (depends on bonds)
        # 4. add proper torsions (depends on 3 bonds and 2 angles)
        # 4. add improper torsions (depends on 3 bonds and 3 angles)

        for (i, j), (force_k, _) in zip(bond.potential.idxs, bond.params):
            if force_k > 0:
                ns = NodeState.ON
            else:
                ns = NodeState.OFF

            bond_node = (NodeType.BOND, (i, j))
            assert not self._dag.has_node(bond_node)
            self._dag.add_node(bond_node, state=ns)

        for (i, j, k), (force_k, _, _) in zip(angle.potential.idxs, angle.params):
            if force_k > 0:
                ns = NodeState.ON
            else:
                ns = NodeState.OFF

            angle_node = (NodeType.ANGLE, (i, j, k))
            assert not self._dag.has_node(angle_node)
            self._dag.add_node(angle_node, state=ns)

            implied_bond_idxs = _get_implied_bond_nodes_from_angle_idxs(i, j, k)
            for bond_node in implied_bond_idxs:
                assert self._dag.has_node(bond_node)
                self._dag.add_edge(angle_node, bond_node)

        for (c, j, k, l), force_k in zip(chiral_atom.potential.idxs, chiral_atom.params):
            if force_k > 0:
                ns = NodeState.ON
            else:
                ns = NodeState.OFF

            chiral_node = (NodeType.CHIRAL_ATOM, (c, j, k, l))
            assert not self._dag.has_node(chiral_node)
            self._dag.add_node(chiral_node, state=ns)

            implied_bond_nodes = _get_implied_bond_nodes_from_chiral_atoms(c, j, k, l)
            for bond_node in implied_bond_nodes:
                assert self._dag.has_node(bond_node)
                self._dag.add_edge(chiral_node, bond_node)

            # add reverse dependencies for angles
            implied_angle_nodes = _get_implied_angle_nodes_from_chiral_atoms(c, j, k, l)
            for angle_node in implied_angle_nodes:
                assert self._dag.has_node(angle_node)
                self._dag.add_edge(angle_node, chiral_node)

            # if ns == NodeState.OFF:
            #     assert angle_on_count < 3
            # elif ns == NodeState.ON:
            #     print("NS ON angle_on_count", angle_on_count)
            # assert angle_on_count == 3

        for (i, j, k, l), (force_k, _, n) in zip(proper_torsion.potential.idxs, proper_torsion.params):
            if force_k > 0:
                ns = NodeState.ON
            else:
                ns = NodeState.OFF

            # need to encode periodicity
            proper_node = (NodeType.PROPER_TORSION, (i, j, k, l, int(n)))
            assert not self._dag.has_node(proper_node)
            self._dag.add_node(proper_node, state=ns)

            implied_angle_nodes = _get_implied_angle_nodes_from_proper_idxs(i, j, k, l)
            for angle_node in implied_angle_nodes:
                assert self._dag.has_node(angle_node)
                self._dag.add_edge(proper_node, angle_node)

            implied_bond_nodes = _get_implied_bond_nodes_from_proper_idxs(i, j, k, l)
            for bond_node in implied_bond_nodes:
                assert self._dag.has_node(bond_node)
                self._dag.add_edge(proper_node, bond_node)

        for (c, j, k, l), (force_k, _, _) in zip(improper_torsion.potential.idxs, improper_torsion.params):
            if force_k > 0:
                ns = NodeState.ON
            else:
                ns = NodeState.OFF

            improper_node = (NodeType.IMPROPER_TORSION, (c, j, k, l))
            assert not self._dag.has_node(improper_node)
            self._dag.add_node(improper_node, state=ns)

            implied_angle_nodes = _get_implied_angle_nodes_from_improper_idxs(c, j, k, l)
            for angle_node in implied_angle_nodes:
                assert self._dag.has_node(angle_node)
                self._dag.add_edge(improper_node, angle_node)

            implied_bond_nodes = _get_implied_bond_nodes_from_improper_idxs(c, j, k, l)
            for bond_node in implied_bond_nodes:
                assert self._dag.has_node(bond_node)
                self._dag.add_edge(improper_node, bond_node)

        self._verify_graph()

    def _verify_angles(self):
        for (node_type, angle_idxs), data in self._dag.nodes(data=True):
            if node_type == NodeType.ANGLE:
                node_state = data["state"]
                if node_state == NodeState.ON:
                    i, j, k = angle_idxs
                    implied_bond_nodes = _get_implied_bond_nodes_from_angle_idxs(i, j, k)
                    for bond_node in implied_bond_nodes:
                        assert self._dag.nodes[bond_node]["state"] == NodeState.ON

    def _verify_propers(self):
        for (node_type, torsion_idxs), data in self._dag.nodes(data=True):
            if node_type == NodeType.PROPER_TORSION:
                node_state = data["state"]
                if node_state == NodeState.ON:
                    i, j, k, l, _ = torsion_idxs  # last element is periodicity
                    implied_bond_nodes = _get_implied_bond_nodes_from_proper_idxs(i, j, k, l)
                    for bond_node in implied_bond_nodes:
                        assert self._dag.nodes[bond_node]["state"] == NodeState.ON

                    implied_angle_nodes = _get_implied_angle_nodes_from_proper_idxs(i, j, k, l)
                    for angle_node in implied_angle_nodes:
                        assert self._dag.nodes[angle_node]["state"] == NodeState.ON

    def _verify_impropers(self):
        for (node_type, torsion_idxs), data in self._dag.nodes(data=True):
            if node_type == NodeType.IMPROPER_TORSION:
                node_state = data["state"]
                if node_state == NodeState.ON:
                    c, j, k, l, _ = torsion_idxs  # last element is periodicity
                    implied_bond_nodes = _get_implied_bond_nodes_from_improper_idxs(c, j, k, l)
                    for bond_node in implied_bond_nodes:
                        assert self._dag.nodes[bond_node]["state"] == NodeState.ON

                    implied_angle_nodes = _get_implied_angle_nodes_from_improper_idxs(c, j, k, l)
                    for angle_node in implied_angle_nodes:
                        assert self._dag.nodes[angle_node]["state"] == NodeState.ON

    def _verify_chiral_atoms(self):
        for (node_type, chiral_idxs), data in self._dag.nodes(data=True):
            if node_type == NodeType.CHIRAL_ATOM:
                node_state = data["state"]
                c, j, k, l = chiral_idxs  # last element is periodicity
                if node_state == NodeState.ON:
                    implied_bond_nodes = _get_implied_bond_nodes_from_chiral_atoms(c, j, k, l)
                    for bond_node in implied_bond_nodes:
                        assert self._dag.nodes[bond_node]["state"] == NodeState.ON
                elif node_state == NodeState.OFF:
                    # if the chiral volume is turned off, we want to avoid a situation where all three
                    # incident chiral angles are turned on, as it would create a slow, trapped, metastable state
                    # that is difficult to sample using conventional MD.
                    implied_angle_nodes = _get_implied_angle_nodes_from_chiral_atoms(c, j, k, l)
                    angle_nodes_on = 0
                    for angle_node in implied_angle_nodes:
                        if self._dag.nodes[angle_node]["state"] == NodeState.ON:
                            angle_nodes_on += 1

                    if angle_nodes_on == 3:
                        print("WARNING: Chiral Idxs", chiral_idxs, "is turned off but all the angles are turned on.")
                    # assert angle_nodes_on < 3

    def _verify_graph(self):
        """
        Check that angles, propers, impropers, chiral atoms have their dependencies satisfied.
        """

        # 1. Angle (i,j,k):
        #   If ON then (i,j) and (j,k) bond states are ON
        #   If OFF then no requirements
        # 2. Proper Torsion (i,j,k,l)
        #   If ON then bonds {(i,j), (j,k), (k,l)} and angles {(i,j,k), (j,k,l)} are ON
        #   If OFF then no requirements
        # 3. Improper Torsion (c,j,k,l)
        #   If ON then bonds {(c,j), (j,k), (k,l)} and angles {(j,c,k), (j,c,l), (k,c,l)} are ON
        #   If OFF then no requirements
        # 4. Chiral Atom (c,j,k,l)
        #   If ON then bonds {(c,j), (c,k), (c,l)} are ON, no requirement on angles.
        #   If OFF then at _most_ 2 of the angles {(j,c,k), (j,c,l), (k,c,l)} are ON
        # 5. Chiral Bond (i,j,k,l)
        #   TBD.

        self._verify_angles()
        self._verify_propers()
        self._verify_impropers()
        self._verify_chiral_atoms()

    def _n_terms(self, state):
        n_bonds = 0
        n_angles = 0
        n_propers = 0
        n_impropers = 0
        n_chiral_atoms = 0

        for node, data in self._dag.nodes(data=True):
            node_type, _ = node
            if node_type == NodeType.BOND:
                if data["state"] == state:
                    n_bonds += 1
            elif node_type == NodeType.ANGLE:
                if data["state"] == state:
                    n_angles += 1
            elif node_type == NodeType.PROPER_TORSION:
                if data["state"] == state:
                    n_propers += 1
            elif node_type == NodeType.IMPROPER_TORSION:
                if data["state"] == state:
                    n_impropers += 1
            elif node_type == NodeType.CHIRAL_ATOM:
                if data["state"] == state:
                    n_chiral_atoms += 1

        return n_bonds, n_angles, n_propers, n_impropers, n_chiral_atoms

    def n_terms_on(self):
        return self._n_terms(state=NodeState.ON)

    def n_terms_off(self):
        return self._n_terms(state=NodeState.OFF)

    def _get_nodes(self, state):
        nodes = set()
        for node, data in self._dag.nodes(data=True):
            if data["state"] == state:
                nodes.add(node)
        return nodes

    def get_on_nodes(self):
        return self._get_nodes(NodeState.ON)

    def get_off_nodes(self):
        return self._get_nodes(NodeState.OFF)


def find_nodes_to_turn_on(lhs_dg, rhs_dg):
    return lhs_dg.get_off_nodes().intersection(rhs_dg.get_on_nodes())


def find_nodes_to_turn_off(lhs_dg, rhs_dg):
    return lhs_dg.get_on_nodes().intersection(rhs_dg.get_off_nodes())


# goal: go from lhs to rhs with a sequence of intermediate dgs that are all "verified" to be valid
#  sequential: greedy strategy, turn on one node at a time
# # reoptimizing into a parallel strategy?
#  parallel: greedy strategy, turn on multiple nodes?
