import copy
from enum import IntEnum

import networkx as nx
from rdkit import Chem
from rdkit.Chem import BondType as BT
from rdkit.Chem import HybridizationType as HT
from rdkit.Chem.Draw import rdMolDraw2D


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
        self.nxg = nx.Graph()
        for atom in mol.GetAtoms():
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

        non_terminal_atoms = []
        for n in self.nxg.nodes():
            nbs = list(self.nxg.neighbors(n))
            if len(nbs) > 1:
                non_terminal_atoms.append(n)

        self.non_terminal_atoms = non_terminal_atoms
        self.reduced_nxg = self.nxg.subgraph(non_terminal_atoms)

    def find_allowed_atom_edits(self):
        # prioritize non-terminal atom deletions
        atoms = []
        for n in self.reduced_nxg.nodes():
            if (n in self.dummy_atoms) and (self.reduced_nxg.degree(n) == 1):
                atoms.append(n)

        # stub dummy atoms next
        for n in self.nxg.nodes():
            if (n in self.dummy_atoms) and (self.nxg.degree(n) == 1) and (n not in atoms):
                atoms.append(n)

        return atoms

    def find_allowed_bond_edits(self):
        bonds = []
        bridges = nx.bridges(self.reduced_nxg)

        # add dummy-dummy non-bridges first
        for src_idx, dst_idx in self.reduced_nxg.edges():
            if src_idx in self.dummy_atoms and dst_idx in self.dummy_atoms:
                if (src_idx, dst_idx) not in bridges and (dst_idx, src_idx) not in bridges:
                    bonds.append((src_idx, dst_idx))

        # add dummy-core non-bridges second
        for src_idx, dst_idx in self.reduced_nxg.edges():
            if (src_idx in self.dummy_atoms and dst_idx in self.core_atoms) or (
                src_idx in self.core_atoms and dst_idx in self.dummy_atoms
            ):
                if (src_idx, dst_idx) not in bridges and (dst_idx, src_idx) not in bridges:
                    bonds.append((src_idx, dst_idx))

        # prefer dummy-dummy non-bridges over core-dummy non-bridges
        return bonds

    def turn_atom_into_dummy(self, idx):
        new_atom_primitives = copy.copy(self.atom_primitives)
        new_atom_states = copy.copy(self.atom_states)
        new_bond_states = copy.copy(self.bond_states)

        new_atom_states[idx] = AtomState.NON_INTERACTING

        for nb in self.nxg.neighbors(idx):
            # implied
            # new_bond_states[self.mol.GetBondBetweenAtoms(idx, nb).GetIdx()] = BondState.
            if len(list(self.nxg.neighbors(nb))) == 1:
                new_atom_states[nb] = AtomState.NON_INTERACTING
            else:
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
            # if bond_state == BondState.DUMMY:
            # bond.SetBondType(BT.ZERO)
            if bond_state == BondState.NON_INTERACTING:
                # bond.SetBondType(BT.ZERO)
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
        for c_a, c_b in zip(self.core_atoms, gp_b.core_atoms):
            print(self.atom_primitives[c_a], "vs", gp_b.atom_primitives[c_b])
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
