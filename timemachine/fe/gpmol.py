import copy
from enum import IntEnum

import networkx as nx
from rdkit import Chem
from rdkit.Chem import BondType as BT
from rdkit.Chem import HybridizationType as HT
from rdkit.Chem.Draw import rdMolDraw2D

# from rdkit.Chem.Draw import rdMolDraw2D1
from timemachine.graph_utils import convert_to_nx


class AtomState(IntEnum):
    NON_INTERACTING = 0
    INTERACTING = 1


class BondState(IntEnum):
    NON_INTERACTING = 0
    INTERACTING = 1


class AtomPrimitive(IntEnum):
    X4_SP3 = 0  # standard methyl
    X3_SP3 = 1  # eg: NH3, can be chiral or achiral
    X3_SP2 = 3  # planar
    X2_KINK = 4  # X-O-X
    X2_LINEAR = 5  # -C#N, c=C=C
    X1 = 6  # -H


def downgrade_atom_primitive(x):
    if x == AtomPrimitive.X4_SP3:
        return AtomPrimitive.X3_SP3
    elif x == AtomPrimitive.X3_SP3:
        return AtomPrimitive.X2_KINK
    elif x == AtomPrimitive.X3_SP2:
        return AtomPrimitive.X2_KINK
    elif x == AtomPrimitive.X2_KINK:
        return AtomPrimitive.X1
    elif x == AtomPrimitive.X2_LINEAR:
        return AtomPrimitive.X1
    elif x == AtomPrimitive.X1:
        assert 0


def initialize_atom_primitives(mol):
    atom_primitives = []

    for atom in mol.GetAtoms():
        hyb = atom.GetHybridization()
        if atom.GetTotalDegree() == 4:
            atom_primitives.append(AtomPrimitive.X4_SP3)
        elif atom.GetTotalDegree() == 3:
            if hyb == HT.SP3:
                atom_primitives.append(AtomPrimitive.X3_SP3)
            elif hyb == HT.SP2:
                atom_primitives.append(AtomPrimitive.X3_SP2)
            else:
                assert 0
        elif atom.GetTotalDegree() == 2:
            if hyb == HT.SP3 or hyb == HT.SP2:
                atom_primitives.append(AtomPrimitive.X2_KINK)
            elif hyb == HT.SP:
                atom_primitives.append(AtomPrimitive.X2_LINEAR)
            else:
                print("UNKNOWN", hyb)
                assert 0
        elif atom.GetTotalDegree() == 1:
            atom_primitives.append(AtomPrimitive.X1)
        else:
            assert 0

    return atom_primitives


class GPMol:
    def __init__(self, mol, core_atoms, atom_primitives, atom_states, bond_states):
        self.mol = mol
        self.core_atoms = core_atoms
        self.dummy_atoms = []
        self.atom_primitives = atom_primitives
        self.atom_states = atom_states
        self.bond_states = bond_states

        # nxg is a subgraph comprised only of interacting atoms
        self.nxg = nx.Graph()
        for atom in mol.GetAtoms():
            atom.SetProp("atomLabel", f"{atom.GetIdx()}")
            atom_idx = atom.GetIdx()
            if atom_idx not in core_atoms:
                self.dummy_atoms.append(atom_idx)
            if atom_states[atom_idx] == AtomState.INTERACTING:
                self.nxg.add_node(atom_idx, primitive=atom_primitives[atom_idx])

        for bond in mol.GetBonds():
            if bond_states[bond.GetIdx()] == BondState.INTERACTING:
                src_idx = bond.GetBeginAtomIdx()
                dst_idx = bond.GetEndAtomIdx()
                if (
                    self.atom_states[src_idx] == AtomState.INTERACTING
                    and self.atom_states[dst_idx] == AtomState.INTERACTING
                ):
                    self.nxg.add_edge(src_idx, dst_idx)

        reduced_graph_atoms = set()
        for n in self.nxg.nodes():
            nbs = list(self.nxg.neighbors(n))
            if len(nbs) > 1 or n in self.core_atoms:
                reduced_graph_atoms.add(n)

        self.reduced_nxg = self.nxg.subgraph(reduced_graph_atoms)

    # test idea: ensure that we can always delete down to the core and back up.
    def find_allowed_atom_deletions(self):
        # delete atoms in a way such that they are *simply* factorizable first.
        # a simply factorizable sub-group is a set of atoms that can be turned into a dummy state
        # *without* breaking any bond or angle terms.

        # ex 1.
        #     C    C           C    D
        #     |   /            |   /
        #   R-C--C--C   -->  --C--C--D
        #     |   \            |   \
        #     C    C           C    D

        # ex 2.
        #      C==C             D==D
        #     /    \           /    \
        #  R=C      C   --> R=C      D
        #     \\  //           \\  //
        #      C--C             D--D
        #

        # v1: only allow deletions on dummy groups that are connected to a single anchor.

        atom_groups = []

        for n in self.reduced_nxg.nodes():
            # pick a terminal atom in the reduced graph
            atoms = []
            # if self.reduced_nxg.degree(n) == 1:
            for nb in self.nxg.neighbors(n):
                if nb in self.dummy_atoms and self.nxg.degree(nb) == 1:
                    atoms.append(nb)
            if len(atoms) > 0:
                atom_groups.append(atoms)

        # print("atom_groups", atom_groups)

        return atom_groups

    # def find_allowed_atom_insertions(self):
    # inversion of the above?
    #     return atoms

    def find_dummy_groups(self):
        dummy_subgraph = nx.Graph()
        for src, dst in self.nxg.edges():
            if src in self.dummy_atoms or dst in self.dummy_atoms:
                dummy_subgraph.add_edge(src, dst)
        ccs = nx.connected_components(dummy_subgraph)
        dummys_and_anchors = {}
        for cc in ccs:
            dg = []
            anchors = []
            for atom in cc:
                if atom in self.core_atoms:
                    anchors.append(atom)
                else:
                    dg.append(atom)

            dummys_and_anchors[tuple(dg)] = tuple(anchors)

        return dummys_and_anchors, dummy_subgraph

    def find_allowed_bond_deletions(self):
        all_bonds_to_delete = []
        dummys_and_anchors, dummy_subgraph = self.find_dummy_groups()
        for dummy_atoms, anchors in dummys_and_anchors.items():
            # every dummy group must be anchored to a core anchor
            assert len(anchors) > 0
            # if this dummy group only has one core anchor, then we don't need to do anything
            if len(anchors) == 1:
                continue
            # if we reach this line of code, then we need to delete bonds and split the dummy group
            # and anchor
            sg = dummy_subgraph.subgraph(dummy_atoms + anchors)
            bonds_to_delete = []
            for src, dst in sg.edges():
                one_bond_deletion_graph = sg.copy()
                one_bond_deletion_graph.remove_edge(src, dst)
                dummys_and_anchors = {}
                for cc in nx.connected_components(one_bond_deletion_graph):
                    dg = []
                    anchors = []
                    for atom in cc:
                        if atom in self.core_atoms:
                            anchors.append(atom)
                        else:
                            dg.append(atom)
                    dummys_and_anchors[tuple(dg)] = tuple(anchors)
                if np.all([len(anchors) == 1 for anchors in dummys_and_anchors.values()]):
                    bonds_to_delete.append((src, dst))

            if len(bonds_to_delete) == 0:
                # (ytz):
                # we need to recursively split each subgroup even more, if so, we will need to return list of bonds
                # that we can delete (as opposed to singleton bonds)
                assert 0

            all_bonds_to_delete.extend(bonds_to_delete)

        return all_bonds_to_delete

    # def find_allowed_bond_deletions(self):
    #     """
    #     Find bonds that can be deleted to satisfy factorizability considerations.
    #     """
    #     bonds = []
    #     bridges = nx.bridges(self.nxg)

    #     ranks = []
    #     for src_idx, dst_idx in self.nxg.edges():
    #         src_is_dummy = src_idx in self.dummy_atoms
    #         dst_is_dummy = dst_idx in self.dummy_atoms
    #         if src_is_dummy or dst_is_dummy:
    #             if (src_idx, dst_idx) not in bridges and (dst_idx, src_idx) not in bridges:

    #                 # totally arbitrary, can be removed later when we do network-level
    #                 # cost network analysis. for now, we arbitrarily prefer breaking ring
    #                 # bonds in the middle vs in the middle of the chain. b
    #                 if src_is_dummy and dst_is_dummy:
    #                     ranks.append(0)
    #                 else:
    #                     ranks.append(1)

    #                 bonds.append((src_idx, dst_idx))

    #     # prefer breaking dummy-dummy over dummy-core bonds
    #     perm = np.argsort(ranks)

    #     return np.array(bonds)[perm].tolist()

    def turn_atoms_into_dummy(self, atom_group):
        new_atom_primitives = copy.copy(self.atom_primitives)
        new_atom_states = copy.copy(self.atom_states)
        new_bond_states = copy.copy(self.bond_states)

        for idx in atom_group:
            new_atom_states[idx] = AtomState.NON_INTERACTING

            for nb in self.nxg.neighbors(idx):
                # also turn all nearby dangling atoms into a dummy.
                # if len(list(self.nxg.neighbors(nb))) == 1:
                # new_atom_states[nb] = AtomState.NON_INTERACTING
                # else:
                new_atom_primitives[nb] = downgrade_atom_primitive(self.atom_primitives[nb])

        return GPMol(self.mol, self.core_atoms, new_atom_primitives, new_atom_states, new_bond_states)

    def delete_bond(self, src_idx, dst_idx):
        new_atom_primitives = copy.copy(self.atom_primitives)
        new_atom_states = copy.copy(self.atom_states)
        new_bond_states = copy.copy(self.bond_states)
        new_bond_states[self.mol.GetBondBetweenAtoms(src_idx, dst_idx).GetIdx()] = BondState.NON_INTERACTING
        new_atom_primitives[src_idx] = downgrade_atom_primitive(self.atom_primitives[src_idx])
        new_atom_primitives[dst_idx] = downgrade_atom_primitive(self.atom_primitives[dst_idx])

        return GPMol(self.mol, self.core_atoms, new_atom_primitives, new_atom_states, new_bond_states)

    def induced_mol(self):
        mol_copy = Chem.RWMol(self.mol)
        Chem.Kekulize(mol_copy, clearAromaticFlags=True)
        for atom in mol_copy.GetAtoms():
            if self.atom_states[atom.GetIdx()] == AtomState.NON_INTERACTING:
                atom.SetAtomicNum(0)
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

        mol_copy.BeginBatchEdit()
        for bond_idx, bond_state in enumerate(self.bond_states):
            bond = mol_copy.GetBondWithIdx(bond_idx)
            if bond_state == BondState.NON_INTERACTING:
                mol_copy.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            else:
                src_idx = bond.GetBeginAtomIdx()
                dst_idx = bond.GetEndAtomIdx()
                src_is_dummy = self.atom_states[src_idx] == AtomState.NON_INTERACTING
                dst_is_dummy = self.atom_states[dst_idx] == AtomState.NON_INTERACTING
                if src_is_dummy or dst_is_dummy:
                    bond.SetBondType(BT.ZERO)

        mol_copy.CommitBatchEdit()
        return mol_copy

    def draw_mol(self):
        drawer = rdMolDraw2D.MolDraw2DSVG(400, 200)
        drawer.DrawMolecule(self.induced_mol())
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    def find_allowed_core_mutations(self, gp_b):
        atoms = []

        # (ytz): TODO: only allow if valence rules are met
        # (dummy atoms are all non-interacting to avoid hyper valency)
        for c_a, c_b in zip(self.core_atoms, gp_b.core_atoms):
            # print(self.atom_primitives[c_a], "vs", gp_b.atom_primitives[c_b])
            if self.atom_primitives[c_a] != gp_b.atom_primitives[c_b]:
                atoms.append(c_a)
        return atoms

    def mutate_atom(self, atom_idx_in_a, gp_b):
        a_to_b_core_map = dict()
        for c_a, c_b in zip(self.core_atoms, gp_b.core_atoms):
            a_to_b_core_map[c_a] = c_b
        atom_idx_in_b = a_to_b_core_map[atom_idx_in_a]

        new_gp_a = copy.deepcopy(self)
        new_gp_a.atom_primitives[atom_idx_in_a] = gp_b.atom_primitives[atom_idx_in_b]

        # update coordinates
        # new_pos = gp_b.mol.GetConformer(0).GetAtomPosition(int(atom_idx_in_b))
        # new_gp_a.mol.GetConformer(0).SetAtomPosition(int(atom_idx_in_a), new_pos)
        return new_gp_a


import numpy as np

from timemachine.fe.single_topology import AtomMapMixin


def _add_mol_to_graph(nxg, mol, mapping):
    for atom in mol.GetAtoms():
        nxg.add_node(mapping[atom.GetIdx()])

    for bond in mol.GetBonds():
        src_idx = bond.GetBeginAtomIdx()
        dst_idx = bond.GetEndAtomIdx()
        nxg.add_edge(mapping[src_idx], mapping[dst_idx])


class ComposedGPMol:
    def __init__(self, gp_a, gp_b):
        self.gp_a = gp_a
        self.gp_b = gp_b

        core = np.array([[x, y] for x, y in zip(gp_a.core_atoms, gp_b.core_Atoms)])

        amm = AtomMapMixin(gp_a.mol, gp_b.mol, core)
        self.amm = amm

        self.nxg = nx.Graph()

        _add_mol_to_graph(self.nxg, gp_a, amm.a_to_c)
        _add_mol_to_graph(self.nxg, gp_b, amm.b_to_c)

    def get_atom_states(self):
        # compute an atom_state composed from individual atom_states.
        combined_state = np.zeros(self.amm.get_num_atoms()) - 1  # initialize to garbage
        for atom_idx, state in enumerate(self.gp_a.atom_states):
            if atom_idx in self.gp_a.core_atoms:
                assert state == AtomState.INTERACTING
            combined_state[self.amm.a_to_c[atom_idx]] = state

        for atom_idx, state in enumerate(self.gp_b.atom_states):
            if atom_idx in self.gp_b.core_atoms:
                assert state == AtomState.INTERACTING

            old_state = combined_state[self.amm.b_to_c[atom_idx]]
            if old_state != -1:
                state == old_state
            else:
                combined_state[self.amm.b_to_c[atom_idx]] = state

        # every atom should be in either one state or the other.
        assert np.any(combined_state == -1) is False

        return combined_state

    def find_allowed_atom_deletions(self):
        # mol_a's atoms can be deleted
        # i.e. mol_a's atom_states can only go from INTERACTING -> NON-INTERACTING
        # and mol_b's atom_states can only go from NON-INTERACTING -> INTERACTING

        # mol_b's atoms can be inserted.
        return atoms

    def find_allowed_atom_deletions(self):
        pass

    def find_allowed_bond_insertions(self):
        pass

    def find_allowed_bond_deletions(self):
        pass
