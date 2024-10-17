from enum import IntEnum

from networkx import DiGraph

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


class DepGraph:
    def __init__(self, bond, angle, proper_torsion, improper_torsion, chiral_atom):
        # self.potentials = potentials
        self._dag = DiGraph()

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

            implied_bond_idxs = [canonicalize_bond([i, j]), canonicalize_bond([j, k])]
            for bond_idxs in implied_bond_idxs:
                bond_node = (NodeType.BOND, tuple(bond_idxs))
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

            for x in [j, k, l]:
                src, dst = canonicalize_bond((c, x))
                bond_node = (NodeType.BOND, (src, dst))
                assert self._dag.has_node(bond_node)
                self._dag.add_edge(chiral_node, bond_node)

            # add reverse dependencies for angles
            implied_angle_idxs = [
                canonicalize_bond([j, c, k]),
                canonicalize_bond([j, c, l]),
                canonicalize_bond([k, c, l]),
            ]
            for angle_idxs in implied_angle_idxs:
                angle_node = (NodeType.ANGLE, tuple(angle_idxs))
                assert self._dag.has_node(angle_node)
                self._dag.add_edge(angle_node, chiral_node)

        for (i, j, k, l), (force_k, _, n) in zip(proper_torsion.potential.idxs, proper_torsion.params):
            if force_k > 0:
                ns = NodeState.ON
            else:
                ns = NodeState.OFF

            # need to encode periodicity
            proper_node = (NodeType.PROPER_TORSION, (i, j, k, l, int(n)))
            assert not self._dag.has_node(proper_node)
            self._dag.add_node(proper_node, state=ns)

            # add reverse dependencies for angles
            implied_angle_idxs = [
                canonicalize_bond([i, j, k]),
                canonicalize_bond([j, k, l]),
            ]

            for angle_idxs in implied_angle_idxs:
                angle_node = (NodeType.ANGLE, tuple(angle_idxs))
                assert self._dag.has_node(angle_node)
                self._dag.add_edge(proper_node, angle_node)

            implied_bond_idxs = [canonicalize_bond([i, j]), canonicalize_bond([j, k]), canonicalize_bond([k, l])]
            for bond_idxs in implied_bond_idxs:
                bond_node = (NodeType.BOND, tuple(bond_idxs))
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

            # add reverse dependencies for angles
            implied_angle_idxs = [
                canonicalize_bond([j, c, k]),
                canonicalize_bond([j, c, l]),
                canonicalize_bond([k, c, l]),
            ]

            for angle_idxs in implied_angle_idxs:
                angle_node = (NodeType.ANGLE, tuple(angle_idxs))
                assert self._dag.has_node(angle_node)
                self._dag.add_edge(improper_node, angle_node)

            implied_bond_idxs = [canonicalize_bond([c, j]), canonicalize_bond([c, k]), canonicalize_bond([c, l])]
            for bond_idxs in implied_bond_idxs:
                bond_node = (NodeType.BOND, tuple(bond_idxs))
                assert self._dag.has_node(bond_node)
                self._dag.add_edge(improper_node, bond_node)

        for src_node, dst_node in self._dag.edges():
            src_state = self._dag.nodes[src_node]["state"]
            dst_state = self._dag.nodes[dst_node]["state"]

            # invalid state
            if src_state == NodeState.ON and dst_state == NodeState.OFF:
                print(src_node, src_state, "->", dst_node, dst_state)
                # assert 0

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
