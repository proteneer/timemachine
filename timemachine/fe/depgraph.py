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

            self._dag.add_node((NodeType.BOND, (i, j)), state=ns)

        for (i, j, k), (force_k, _, _) in zip(angle.potential.idxs, angle.params):
            if force_k > 0:
                ns = NodeState.ON
            else:
                ns = NodeState.OFF

            angle_node = (NodeType.ANGLE, (i, j, k))
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

        for (i, j, k, l), (force_k, _, _) in zip(proper_torsion.potential.idxs, proper_torsion.params):
            if force_k > 0:
                ns = NodeState.ON
            else:
                ns = NodeState.OFF

            torsion_node = (NodeType.PROPER_TORSION, (i, j, k, l))
            self._dag.add_node(torsion_node, state=ns)

            # add reverse dependencies for angles
            implied_angle_idxs = [
                canonicalize_bond([i, j, k]),
                canonicalize_bond([j, k, l]),
            ]
            for angle_idxs in implied_angle_idxs:
                angle_node = (NodeType.ANGLE, tuple(angle_idxs))
                assert self._dag.has_node(angle_node)
                self._dag.add_edge(angle_node, chiral_node)

            implied_bond_idxs = [canonicalize_bond([i, j]), canonicalize_bond([j, k]), canonicalize_bond([k, l])]
            for bond_idxs in implied_bond_idxs:
                bond_node = (NodeType.BOND, tuple(bond_idxs))
                assert self._dag.has_node(bond_node)
                self._dag.add_edge(angle_node, bond_node)

        for (c, j, k, l), (force_k, _, _) in zip(improper_torsion.potential.idxs, improper_torsion.params):
            if force_k > 0:
                ns = NodeState.ON
            else:
                ns = NodeState.OFF

            torsion_node = (NodeType.IMPROPER_TORSION, (c, j, k, l))
            self._dag.add_node(torsion_node, state=ns)

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

            implied_bond_idxs = [canonicalize_bond([c, j]), canonicalize_bond([c, k]), canonicalize_bond([c, l])]
            for bond_idxs in implied_bond_idxs:
                bond_node = (NodeType.BOND, tuple(bond_idxs))
                assert self._dag.has_node(bond_node)
                self._dag.add_edge(angle_node, bond_node)
