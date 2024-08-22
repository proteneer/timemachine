import copy
from enum import IntEnum

import networkx as nx
from rdkit import Chem
from rdkit.Chem import BondType as BT
from rdkit.Chem import HybridizationType as HT
from rdkit.Chem.Draw import rdMolDraw2D

from timemachine.fe.topology import BaseTopology


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
            # atom.SetProp("atomLabel", f"{atom.GetIdx()}")
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

    def get_bond_state(self, src, dst):
        bond = self.mol.GetBondBetweenAtoms(int(src), int(dst))
        return self.bond_states[bond.GetIdx()]

    def get_atom_state(self, src):
        return self.atom_states[src]

    def find_anchor_dummy_atom_deletions(self):
        # delete dummy atoms from the anchor. no guarantees are made about factorizability.
        dummys_and_anchors, _ = self.find_dummy_groups()
        atom_groups = []
        for k, v in dummys_and_anchors.items():
            if len(v) == 1:
                atom_groups.append(k)

        return atom_groups

    def find_simply_factorizable_atom_deletions(self):
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
        atom_groups = []
        # find bridges in the reduced graph
        # recall that reduced graph truncates the dummy subgraph, but *fully* preserves
        # the core graph.
        for src, dst in nx.bridges(self.reduced_nxg):
            # a bridge bond, upon deletion, results in a disconnected graph, forming two node
            # partitions:

            # 1. one partition contains core atoms, the other partition contains only dummy atoms.
            # 2. both partitions contain core atoms (eg. if bridge exists in the core)
            graph_copy = self.nxg.copy()
            graph_copy.remove_edge(src, dst)

            ccs = list(nx.connected_components(graph_copy))
            assert len(ccs) == 2

            dummy_only_cc = None
            for cc in ccs:
                # remove bridge atoms from the connected components
                cc_truncated = [x for x in cc if (x != src and x != dst)]
                if np.all([x in self.dummy_atoms for x in cc_truncated]):
                    assert dummy_only_cc is None
                    dummy_only_cc = cc_truncated

            if dummy_only_cc:
                atom_groups.append(dummy_only_cc)

        atom_groups.sort(key=len)

        return atom_groups

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

        # prefer dummy-dummy bonds to dummy-core bonds
        def bond_score(bond):
            src, dst = bond
            if src in self.dummy_atoms and dst in self.dummy_atoms:
                return 0
            else:
                return 1

        all_bonds_to_delete.sort(key=bond_score)

        return all_bonds_to_delete

    def turn_atoms_into_dummy(self, atom_group):
        new_atom_primitives = copy.copy(self.atom_primitives)
        new_atom_states = copy.copy(self.atom_states)
        new_bond_states = copy.copy(self.bond_states)

        for idx in atom_group:
            new_atom_states[idx] = AtomState.NON_INTERACTING
            for nb in self.nxg.neighbors(idx):
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

    def find_allowed_core_geometry_mutations(self, gp_b):
        atoms = []

        # (ytz): TODO: only allow if valence rules are met
        # (dummy atoms are all non-interacting to avoid hyper valency)
        for c_a, c_b in zip(self.core_atoms, gp_b.core_atoms):
            if self.atom_primitives[c_a] != gp_b.atom_primitives[c_b]:
                atoms.append(c_a)

        return atoms

    def mutate_core_atom(self, atom_idx_in_a, gp_b):
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


def initialize_mol_to_gp(mol, core_atoms):
    atom_primitives = initialize_atom_primitives(mol)
    atom_states = np.array([AtomState.INTERACTING for _ in range(mol.GetNumAtoms())])
    bond_states = np.array([BondState.INTERACTING for _ in range(mol.GetNumBonds())])
    return GPMol(mol, core_atoms, atom_primitives, atom_states, bond_states)


from timemachine.fe.single_topology import canonicalize_bond, canonicalize_improper_idxs
from timemachine.fe.system import VacuumSystem
from timemachine.potentials import PeriodicTorsion


def get_atom_states(gp_a, gp_b, amm):
    # compute an atom_state composed from individual atom_states.
    combined_state = np.zeros(amm.get_num_atoms(), dtype=np.int32) - 1  # initialize to garbage
    for atom_idx, state in enumerate(gp_a.atom_states):
        if atom_idx in gp_a.core_atoms:
            assert state == AtomState.INTERACTING
        combined_state[amm.a_to_c[atom_idx]] = state

    for atom_idx, state in enumerate(gp_b.atom_states):
        if atom_idx in gp_b.core_atoms:
            assert state == AtomState.INTERACTING

        old_state = combined_state[amm.b_to_c[atom_idx]]
        if old_state != -1:
            state == old_state
        else:
            combined_state[amm.b_to_c[atom_idx]] = state

    # print(gp_a.atom_states)
    # print(gp_b.atom_states)
    # print(combined_state)

    # every atom should be in either one state or the other.
    assert np.all([x != -1 for x in combined_state])

    return combined_state


def induce_parameters(gp, bt, ff, a_to_c):
    bond_params, hb = bt.parameterize_harmonic_bond(ff.hb_handle.params)
    angle_params, ha = bt.parameterize_harmonic_angle(ff.ha_handle.params)
    proper_params, pt = bt.parameterize_proper_torsion(ff.pt_handle.params)
    improper_params, it = bt.parameterize_improper_torsion(ff.it_handle.params)
    nbpl_params, nbpl = bt.parameterize_nonbonded_pairlist(
        ff.q_handle.params,
        ff.q_handle_intra.params,
        ff.lj_handle.params,
        ff.lj_handle_intra.params,
        intramol_params=True,
    )

    for params, idxs in zip(bond_params, hb.idxs):
        src, dst = idxs
        if gp.get_bond_state(src, dst) == BondState.NON_INTERACTING:
            params[0] = 0
    hb.idxs = a_to_c[hb.idxs]
    hb.idxs = np.array([canonicalize_bond(x) for x in hb.idxs], dtype=np.int32)
    hb = hb.bind(bond_params)

    for params, idxs in zip(angle_params, ha.idxs):
        src, mid, dst = idxs
        if (
            gp.get_bond_state(src, mid) == BondState.NON_INTERACTING
            or gp.get_bond_state(mid, dst) == BondState.NON_INTERACTING
        ):
            params[0] = 0
    ha.idxs = a_to_c[ha.idxs]
    ha.idxs = np.array([canonicalize_bond(x) for x in ha.idxs], dtype=np.int32)
    # ha = ha.bind(angle_params)
    stable_angle_params = np.hstack([angle_params, np.zeros((len(angle_params), 1))])
    ha = HarmonicAngleStable(ha.idxs).bind(stable_angle_params)

    for params, idxs in zip(proper_params, pt.idxs):
        if np.any([gp.get_atom_state(x) == AtomState.NON_INTERACTING for x in idxs]):
            params[0] = 0
    pt.idxs = a_to_c[pt.idxs]
    pt.idxs = np.array([canonicalize_bond(x) for x in pt.idxs], dtype=np.int32)

    for params, idxs in zip(improper_params, it.idxs):
        if np.any([gp.get_atom_state(x) == AtomState.NON_INTERACTING for x in idxs]):
            params[0] = 0
    it.idxs = a_to_c[it.idxs]
    it.idxs = np.array([canonicalize_improper_idxs(x) for x in it.idxs], dtype=np.int32)

    torsion_idxs = np.concatenate([pt.idxs, it.idxs])
    torsion_params = np.concatenate([proper_params, improper_params])
    torsion = PeriodicTorsion(torsion_idxs).bind(torsion_params)

    for params, idxs in zip(nbpl_params, nbpl.idxs):
        src, dst = idxs
        if gp.get_atom_state(src) == AtomState.NON_INTERACTING or gp.get_atom_state(dst) == AtomState.NON_INTERACTING:
            # set w-offset to cutoff
            params[-1] = 1.2
    nbpl.idxs = a_to_c[nbpl.idxs]
    nbpl.idxs = np.array([canonicalize_bond(x) for x in nbpl.idxs], dtype=np.int32)
    nbpl = nbpl.bind(nbpl_params)

    return VacuumSystem(hb, ha, torsion, nbpl, None, None)


def make_mol(mol_a, mol_b, core, vs, dir, atom_states, min_bond_k=100.0) -> Chem.Mol:
    """
    Generate an RDKit mol, with the dummy atoms attached to the molecule. Atom types and bond parameters
    guesstimated from the corresponding bond orders.

    Tricky-bits to figure out later on: Inferring bond orders and atom-types.

    Parameters
    ----------
    lamb: float
        Lambda value to use

    min_bond_k: float
        Minimum force constant required for a bond to be present in the mol

    Returns
    -------
    Chem.Mol
    """

    from timemachine.fe.single_topology import AtomMapMixin

    amm = AtomMapMixin(mol_a, mol_b, core)
    N = amm.get_num_atoms()
    mol_a_atomic_nums = [a.GetAtomicNum() for a in mol_a.GetAtoms()]
    mol_b_atomic_nums = [b.GetAtomicNum() for b in mol_b.GetAtoms()]
    mol = Chem.RWMol()

    old_to_new_kv = dict()
    for c_idx in range(N):
        if c_idx in amm.c_to_a and c_idx in amm.c_to_b:
            # core, in both mol_a and mol_b
            if dir == "fwd":
                atomic_num = mol_a_atomic_nums[amm.c_to_a[c_idx]]
            elif dir == "rev":
                atomic_num = mol_b_atomic_nums[amm.c_to_b[c_idx]]
            else:
                assert 0
        elif c_idx in amm.c_to_a:
            # only in mol_a
            atomic_num = mol_a_atomic_nums[amm.c_to_a[c_idx]]
        elif c_idx in amm.c_to_b:
            # only in mol_b
            atomic_num = mol_b_atomic_nums[amm.c_to_b[c_idx]]
        else:
            # in neither, assert
            assert 0

        if atom_states[c_idx] == AtomState.INTERACTING:
            atom = Chem.Atom(atomic_num)
            old_to_new_kv[c_idx] = mol.GetNumAtoms()
            mol.AddAtom(atom)
        else:
            old_to_new_kv[c_idx] = -1

    # print("VS BONDS", vs.bond.potential.idxs)

    # setup bonds
    for (old_i, old_j), (k, b) in zip(vs.bond.potential.idxs, vs.bond.params):
        new_i, new_j = old_to_new_kv[old_i], old_to_new_kv[old_j]

        if new_i != -1 and new_j != -1:
            assert atom_states[old_i] == AtomState.INTERACTING
            assert atom_states[old_j] == AtomState.INTERACTING

            if k > min_bond_k:
                mol.AddBond(int(new_i), int(new_j), Chem.BondType.SINGLE)

    # assert 0
    # make read-only
    return Chem.Mol(mol), old_to_new_kv


def combine_vacuum_systems(vs1, vs2):
    """
    Combine two vacuum systems. If there are duplicate idxs then parameters in vs_1 win.
    """

    vsc = copy.deepcopy(vs1)
    vsc_bonds = set([tuple(x) for x in vsc.bond.potential.idxs])
    extra_bond_idxs = []
    extra_bond_params = []
    for bond, params in zip(vs2.bond.potential.idxs, vs2.bond.params):
        if tuple(bond) not in vsc_bonds:
            extra_bond_idxs.append(bond)
            extra_bond_params.append(params)

    vsc.bond.potential.idxs = np.concatenate([vsc.bond.potential.idxs, np.array(extra_bond_idxs)])
    vsc.bond.params = np.concatenate([vsc.bond.params, np.array(extra_bond_params)])

    vsc_angles = set([tuple(x) for x in vsc.angle.potential.idxs])
    extra_angle_idxs = []
    extra_angle_params = []
    for angle, params in zip(vs2.angle.potential.idxs, vs2.angle.params):
        if tuple(angle) not in vsc_angles:
            extra_angle_idxs.append(angle)
            extra_angle_params.append(params)

    vsc.angle.potential.idxs = np.concatenate([vsc.angle.potential.idxs, np.array(extra_angle_idxs)])
    vsc.angle.params = np.concatenate([vsc.angle.params, np.array(extra_angle_params)])

    vsc_torsions = set([tuple(x) for x in vsc.torsion.potential.idxs])
    extra_torsion_idxs = []
    extra_torsion_params = []
    for torsion, params in zip(vs2.torsion.potential.idxs, vs2.torsion.params):
        if tuple(torsion) not in vsc_torsions:
            extra_torsion_idxs.append(torsion)
            extra_torsion_params.append(params)

    vsc.torsion.potential.idxs = np.concatenate([vsc.torsion.potential.idxs, np.array(extra_torsion_idxs)])
    vsc.torsion.params = np.concatenate([vsc.torsion.params, np.array(extra_torsion_params)])

    vsc_nonbondeds = set([tuple(x) for x in vsc.nonbonded.potential.idxs])
    extra_nonbonded_idxs = []
    extra_nonbonded_params = []
    for nonbonded, params in zip(vs2.nonbonded.potential.idxs, vs2.nonbonded.params):
        if tuple(nonbonded) not in vsc_nonbondeds:
            extra_nonbonded_idxs.append(nonbonded)
            extra_nonbonded_params.append(params)

    vsc.nonbonded.potential.idxs = np.concatenate([vsc.nonbonded.potential.idxs, np.array(extra_nonbonded_idxs)])
    vsc.nonbonded.params = np.concatenate([vsc.nonbonded.params, np.array(extra_nonbonded_params)])

    return vsc


def generate_chain(mol, core_atoms):
    atom_primitives_a = initialize_atom_primitives(mol)

    atom_states_a = np.array([AtomState.INTERACTING for _ in range(mol.GetNumAtoms())])
    bond_states_a = np.array([BondState.INTERACTING for _ in range(mol.GetNumBonds())])
    gp_a = GPMol(mol, core_atoms, atom_primitives_a, atom_states_a, bond_states_a)

    cur_gp = gp_a
    path_gps = []

    counter = 0
    while True:
        # svg = cur_gp.draw_mol()
        # fpath = f"mol_{counter}.svg"
        # with open(fpath, "w") as fh:
        #     fh.write(svg)

        counter += 1
        path_gps.append(cur_gp)

        new_gp = None
        for atom_group in cur_gp.find_simply_factorizable_atom_deletions():
            new_gp = cur_gp.turn_atoms_into_dummy(atom_group)
            break

        # no atom edits were found
        if new_gp is None:
            for atom_group in cur_gp.find_anchor_dummy_atom_deletions():
                new_gp = cur_gp.turn_atoms_into_dummy(atom_group)
                break

            if new_gp is None:
                for src, dst in cur_gp.find_allowed_bond_deletions():
                    new_gp = cur_gp.delete_bond(src, dst)
                    break

        if new_gp:
            cur_gp = new_gp
        else:
            break

    return path_gps


from functools import partial
from typing import TypeVar, Union, cast

import jax
import jax.numpy as jnp

from timemachine.fe import interpolate
from timemachine.fe.single_topology import (
    interpolate_harmonic_angle_params,
    interpolate_harmonic_bond_params,
    interpolate_periodic_torsion_params,
    interpolate_w_coord,
)
from timemachine.potentials import (
    BoundPotential,
    ChiralAtomRestraint,
    ChiralBondRestraint,
    HarmonicAngleStable,
    HarmonicBond,
    Nonbonded,
    NonbondedPairListPrecomputed,
    PeriodicTorsion,
    SummedPotential,
)

_Bonded = TypeVar("_Bonded", bound=Union[ChiralAtomRestraint, HarmonicAngleStable, HarmonicBond, PeriodicTorsion])


def _setup_intermediate_bonded_term(
    src_bond: BoundPotential[_Bonded], dst_bond: BoundPotential[_Bonded], lamb, align_fn, interpolate_fn
) -> BoundPotential[_Bonded]:
    src_cls_bond = type(src_bond.potential)
    dst_cls_bond = type(dst_bond.potential)

    assert src_cls_bond == dst_cls_bond

    bond_idxs_and_params = align_fn(
        src_bond.potential.idxs,
        src_bond.params,
        dst_bond.potential.idxs,
        dst_bond.params,
    )
    bond_idxs = np.array([x for x, _, _ in bond_idxs_and_params], dtype=np.int32)
    if bond_idxs_and_params:
        src_params = jnp.array([x for _, x, _ in bond_idxs_and_params])
        dst_params = jnp.array([x for _, _, x in bond_idxs_and_params])
        bond_params = jax.vmap(interpolate_fn, (0, 0, None))(src_params, dst_params, lamb)
    else:
        bond_params = jnp.array([])

    r = src_cls_bond(bond_idxs).bind(bond_params)
    return cast(BoundPotential[_Bonded], r)  # unclear why cast is needed for mypy


def _setup_intermediate_nonbonded_term(
    src_nonbonded: BoundPotential[NonbondedPairListPrecomputed],
    dst_nonbonded: BoundPotential[NonbondedPairListPrecomputed],
    lamb,
    align_fn,
    interpolate_qlj_fn,
) -> BoundPotential[NonbondedPairListPrecomputed]:
    assert src_nonbonded.potential.beta == dst_nonbonded.potential.beta
    assert src_nonbonded.potential.cutoff == dst_nonbonded.potential.cutoff

    cutoff = src_nonbonded.potential.cutoff

    pair_idxs_and_params = align_fn(
        src_nonbonded.potential.idxs,
        src_nonbonded.params,
        dst_nonbonded.potential.idxs,
        dst_nonbonded.params,
    )

    pair_idxs = np.array([x for x, _, _ in pair_idxs_and_params], dtype=np.int32)

    if pair_idxs_and_params:
        src_params = jnp.array([x for _, x, _ in pair_idxs_and_params])
        dst_params = jnp.array([x for _, _, x in pair_idxs_and_params])

        src_qlj, src_w = src_params[:, :3], src_params[:, 3]
        dst_qlj, dst_w = dst_params[:, :3], dst_params[:, 3]

        is_excluded_src = jnp.all(src_qlj == 0.0, axis=1, keepdims=True)
        is_excluded_dst = jnp.all(dst_qlj == 0.0, axis=1, keepdims=True)

        # parameters for pairs that do not interact in the src state
        w = interpolate_w_coord(cutoff, dst_w, lamb)
        pair_params_excluded_src = jnp.concatenate((dst_qlj, w[:, None]), axis=1)

        # parameters for pairs that do not interact in the dst state
        w = interpolate_w_coord(src_w, cutoff, lamb)
        pair_params_excluded_dst = jnp.concatenate((src_qlj, w[:, None]), axis=1)

        # parameters for pairs that interact in both src and dst states
        w = jax.vmap(interpolate.linear_interpolation, (0, 0, None))(src_w, dst_w, lamb)
        qlj = interpolate_qlj_fn(src_qlj, dst_qlj, lamb)
        pair_params_not_excluded = jnp.concatenate((qlj, w[:, None]), axis=1)

        pair_params = jnp.where(
            is_excluded_src,
            pair_params_excluded_src,
            jnp.where(
                is_excluded_dst,
                pair_params_excluded_dst,
                pair_params_not_excluded,
            ),
        )
    else:
        pair_params = jnp.array([])

    return NonbondedPairListPrecomputed(pair_idxs, src_nonbonded.potential.beta, src_nonbonded.potential.cutoff).bind(
        pair_params
    )


def setup_intermediate_state_standard(lamb, src_system, dst_system) -> VacuumSystem:
    r"""
    Set up intermediate states at some value of the alchemical parameter :math:`\lambda`.

    Parameters
    ----------
    lamb: float

    Notes
    -----
    For transformations involving formation or deletion of valence terms (i.e., having force constants equal to zero
    in the :math:`\lambda=0` or :math:`\lambda=1` state), harmonic bond and angle terms are activated before
    torsions. This is to avoid a potential numerical instability in the torsion functional form when three atoms are
    collinear.

    - Bonds and angles with :math:`k=0` at :math:`\lambda=0` are activated in the interval :math:`0 \leq \lambda \leq 0.7`
    - Torsions with :math:`k=0` at :math:`\lambda=0` are activated in the interval :math:`0.7 \leq \lambda \leq 1.0`

    (and similarly for terms with :math:`k=0` at :math:`\lambda=0`, taking :math:`\lambda \to 1-\lambda` in the above.)

    Note that the above only applies to the interactions whose force constant is zero in one end state; otherwise,
    valence terms are interpolated simultaneously in the interval :math:`0 \leq \lambda \leq 1`)
    """
    # stagger the lambda schedule
    bonds_min, bonds_max = [0.0, 0.7]
    angles_min, angles_max = [0.0, 0.7]
    torsions_min, torsions_max = [0.7, 1.0]
    # chiral_atoms_min, chiral_atoms_max = [0.7, 1.0]

    bond = _setup_intermediate_bonded_term(
        src_system.bond,
        dst_system.bond,
        lamb,
        interpolate.align_harmonic_bond_idxs_and_params,
        partial(
            interpolate_harmonic_bond_params,
            k_min=0.1,  # ~ BOLTZ * (300 K) / (5 nm)^2
            lambda_min=bonds_min,
            lambda_max=bonds_max,
        ),
    )

    angle = _setup_intermediate_bonded_term(
        src_system.angle,
        dst_system.angle,
        lamb,
        interpolate.align_harmonic_angle_idxs_and_params,
        partial(
            interpolate_harmonic_angle_params,
            k_min=0.05,  # ~ BOLTZ * (300 K) / (2 * pi)^2
            lambda_min=angles_min,
            lambda_max=angles_max,
        ),
    )

    assert src_system.torsion
    assert dst_system.torsion
    torsion = _setup_intermediate_bonded_term(
        src_system.torsion,
        dst_system.torsion,
        lamb,
        interpolate.align_torsion_idxs_and_params,
        partial(interpolate_periodic_torsion_params, lambda_min=torsions_min, lambda_max=torsions_max),
    )

    nonbonded = _setup_intermediate_nonbonded_term(
        src_system.nonbonded,
        dst_system.nonbonded,
        lamb,
        interpolate.align_nonbonded_idxs_and_params,
        interpolate.linear_interpolation,
    )

    # assert src_system.chiral_atom
    # assert dst_system.chiral_atom

    # assert len(set(tuple(x) for x in src_system.chiral_atom.potential.idxs)) == len(
    #     src_system.chiral_atom.potential.idxs
    # )
    # assert len(set(tuple(x) for x in dst_system.chiral_atom.potential.idxs)) == len(
    #     dst_system.chiral_atom.potential.idxs
    # )

    # chiral_atom = self._setup_intermediate_bonded_term(
    #     src_system.chiral_atom,
    #     dst_system.chiral_atom,
    #     lamb,
    #     interpolate.align_chiral_atom_idxs_and_params,
    #     partial(
    #         interpolate_harmonic_force_constant,
    #         k_min=0.025,
    #         lambda_min=chiral_atoms_min,
    #         lambda_max=chiral_atoms_max,
    #     ),
    # )

    # assert src_system.chiral_bond
    # assert dst_system.chiral_bond
    # chiral_bond = self._setup_intermediate_chiral_bond_term(
    #     src_system.chiral_bond,
    #     dst_system.chiral_bond,
    #     lamb,
    #     interpolate.linear_interpolation,
    # )

    return VacuumSystem(bond, angle, torsion, nonbonded, None, None)


class SingleTopologyV5(AtomMapMixin):
    def __init__(self, mol_a, mol_b, core, ff):
        super().__init__(mol_a, mol_b, core)

        self.ff = ff

        # generate chain of deletions
        self.mol_a_path = generate_chain(self.mol_a, core[:, 0])
        self.mol_b_path = generate_chain(self.mol_b, core[:, 1])

        self.mol_c_path_fwd = []

        bt_a = BaseTopology(mol_a, ff)
        bt_b = BaseTopology(mol_b, ff)

        vs_fwd = []
        vs_rev = []

        atom_states_fwd = []
        atom_states_rev = []

        for gp_a in self.mol_a_path:
            vs_a = induce_parameters(gp_a, bt_a, ff, self.a_to_c)
            vs_b = induce_parameters(self.mol_b_path[-1], bt_b, ff, self.b_to_c)
            vs_c = combine_vacuum_systems(vs_a, vs_b)  # direction matters!
            atom_states_fwd.append(get_atom_states(gp_a, self.mol_b_path[-1], self))
            vs_fwd.append(vs_c)

        for gp_b in self.mol_b_path:
            vs_b = induce_parameters(gp_b, bt_b, ff, self.b_to_c)
            vs_a = induce_parameters(self.mol_a_path[-1], bt_a, ff, self.a_to_c)
            vs_c = combine_vacuum_systems(vs_b, vs_a)  # direction matters!
            atom_states_rev.append(get_atom_states(self.mol_a_path[-1], gp_b, self))
            vs_rev.append(vs_c)

        self.fwd_idx = len(atom_states_fwd)
        self.chain_atom_states = atom_states_fwd + atom_states_rev[::-1]
        self.checkpoint_states = vs_fwd + vs_rev[::-1]

        print("number of checkpoint states", len(self.checkpoint_states))

        self.i_mols, self.i_kvs = self.generate_intermediate_mols_and_kvs()

    def generate_intermediate_mols_and_kvs(self):
        i_mols = []
        kvs = []
        for idx, (vs, atom_states) in enumerate(zip(self.checkpoint_states, self.chain_atom_states)):
            if idx < self.fwd_idx:
                dir = "fwd"
            else:
                dir = "rev"
            mol, old_to_new_kv = make_mol(self.mol_a, self.mol_b, self.core, vs, dir, atom_states)
            i_mols.append(mol)
            kvs.append(old_to_new_kv)

        return i_mols, kvs

    def get_checkpoint_lambdas(self):
        return np.linspace(0, 1, len(self.checkpoint_states))

    def setup_intermediate_state(self, lamb):
        checkpoint_schedule = self.get_checkpoint_lambdas()

        # want lamb = 0 to be 0
        # want lamb = 1 to be checkpoint_states[-1],

        loc = lamb * (len(self.checkpoint_states) - 1)
        lhs_idx = int(np.floor(loc))
        rhs_idx = lhs_idx + 1
        # deal with the case of lamb=1.0
        rhs_idx = min(lhs_idx + 1, len(self.checkpoint_states) - 1)

        # loc - checkpoint_schedule[lhs_idx]
        frac_lamb = lamb - checkpoint_schedule[lhs_idx]

        print("setup", frac_lamb, "lhs_idx", lhs_idx, "rhs_idx", rhs_idx)

        assert frac_lamb >= 0.0
        assert frac_lamb <= 1.0

        return setup_intermediate_state_standard(
            frac_lamb, self.checkpoint_states[lhs_idx], self.checkpoint_states[rhs_idx]
        )

    def setup_intermediate_mol_and_kv(self, lamb):
        loc = lamb * (len(self.checkpoint_states) - 1)
        lhs_idx = int(np.floor(loc))
        # deal with the case of lamb=1.0
        rhs_idx = min(lhs_idx + 1, len(self.checkpoint_states) - 1)

        print("lambda", lamb, "indexing into", lhs_idx, rhs_idx)

        if loc < self.fwd_idx:
            return self.i_mols[lhs_idx], self.i_kvs[lhs_idx]
        else:
            return self.i_mols[rhs_idx], self.i_kvs[rhs_idx]
