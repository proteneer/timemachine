#!/usr/bin/env python
# mypy: ignore-errors

# converts smirnoff xmls into python dictionaries.
# example usage:
# python timemachine/ff/amber_converter.py --input_path amber99sbildn.xml --method "nonbonded"

import pprint
from argparse import ArgumentParser
from typing import Any, Dict, List
from xml.dom import minidom

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

RESIDUE_TAG = "Residue"
BOND_TAG = "HarmonicBondForce"
ANGLE_TAG = "HarmonicAngleForce"
PROPER_TAG = "PeriodicTorsionForce"
NONBONDED_TAG = "NonbondedForce"

tags = [RESIDUE_TAG, BOND_TAG, ANGLE_TAG, PROPER_TAG, NONBONDED_TAG]

STANDARD_RESIDUES = [
    "ace",
    "ala",
    "arg",
    "asn",  # asparagine
    "asp",  # aspartic acid
    "cys",
    "cym",  # negatively charged cysteine
    "cyx",  # disulfide cysteine
    "gln",
    "glh",  # glutamic acid (neutral)
    "glu",  # glutamic acid (charged)
    "gly",
    "hip",  # protonated histidine
    "hid",  # first tautomer (delta location)
    "hie",  # second tautomer (epsilon location)
    "ile",
    "leu",
    "lys",
    "met",
    "nme",
    "nmet",
    "phe",
    "pro",
    "ser",
    "thr",
    "trp",
    "tyr",
    "val",
]


def make_residue_mol(atoms, bonds):
    # Generate an rdkit molecule given a list of atoms and a list of bonds
    mw = Chem.RWMol()
    mw.BeginBatchEdit()
    for atom in atoms:
        aa = Chem.Atom(atom)
        mw.AddAtom(aa)

    for src, dst in bonds:
        mw.AddBond(src, dst, Chem.BondType.SINGLE)
    mw.CommitBatchEdit()

    for atom in mw.GetAtoms():
        atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
    return mw


def dual_sort(src_key, dst_key, src_idx, dst_idx):
    # sort (src_key, dst_key) along with (src_idx, dst_idx)
    # this is important for maintaining directionality of the BCCs.
    if src_key < dst_key:
        return src_key, dst_key, src_idx, dst_idx
    else:
        return dst_key, src_key, dst_idx, src_idx


def process_ff(args):
    xmldoc = minidom.parse(args.input_path)

    # process atom types
    itemlist = xmldoc.getElementsByTagName("Type")
    atom_types_name_to_elem = {}
    atom_types_name_to_class = {}
    for atom_info in itemlist:
        name = atom_info.attributes["name"].value
        elem = atom_info.attributes["element"].value
        cls = atom_info.attributes["class"].value
        atom_types_name_to_elem[name] = elem
        atom_types_name_to_class[name] = cls

    # process bond types
    hb_bond_set = set()
    itemlist = xmldoc.getElementsByTagName(BOND_TAG)
    for res in itemlist:
        for idx, cn in enumerate(res.childNodes):
            if cn.nodeName == "Bond":
                c1 = cn.attributes["class1"].value
                c2 = cn.attributes["class2"].value
                c1, c2 = sorted((c1, c2))
                key_tuple = (c1, c2)
                assert key_tuple not in hb_bond_set
                hb_bond_set.add(key_tuple)

    # process nonbonded types
    # the bond-templates under specifies symmetries, so we do one more level
    # of indirection to look up the actual charge, sigma, epsilon values and
    # generate a new atom type hash.
    nb_atom_types_hash = {}
    itemlist = xmldoc.getElementsByTagName(NONBONDED_TAG)
    for res in itemlist:
        for idx, cn in enumerate(res.childNodes):
            if cn.nodeName == "Atom":
                a_type: str = str(cn.attributes["type"].value)
                charge: str = cn.attributes["charge"].value
                sig: str = cn.attributes["sigma"].value
                eps: str = cn.attributes["epsilon"].value
                key: str = charge + "_" + sig + "_" + eps
                assert a_type not in nb_atom_types_hash
                nb_atom_types_hash[a_type] = key

    all_res_mols: Any = []
    all_residues: Any = []
    all_bond_classes: Any = []
    all_bond_idxs: Any = []

    for tag in tags:
        print(tag)

    for tag in tags:
        itemlist = xmldoc.getElementsByTagName(tag)
        if tag == RESIDUE_TAG:
            for res in itemlist:
                res_name = res.attributes["name"].value
                print("==== Processing residue", res_name, "====")
                atom_iota = 0
                atom_symbols = []
                atom_idxs_to_types: Dict[int, str] = {}
                atom_idxs_to_names = {}
                atom_types_to_idxs = {}

                bond_types_to_param_idx: Dict[str, int] = {}
                bond_param_idxs = []
                bond_list = []
                bond_iota = 0

                bond_type_map: Dict[tuple, int] = {}

                for idx, cn in enumerate(res.childNodes):
                    if cn.nodeName == "Atom":
                        atom_type: Any = str(cn.attributes["type"].value)
                        symbol = atom_types_name_to_elem[atom_type]
                        atom_symbols.append(symbol)
                        atom_idxs_to_types[atom_iota] = atom_type
                        atom_idxs_to_names[atom_iota] = cn.attributes["name"].value
                        atom_types_to_idxs[atom_type] = atom_iota
                        atom_iota += 1

                    elif cn.nodeName == "Bond" and args.method == "harmonic_bond":
                        # generate BCCs based on harmonic_bond atom types, note this "over-symmetrizes"
                        # i.e. symmetry classes are too large, and some just look grossly wrong, eg. serine
                        src_idx = int(cn.attributes["from"].value)
                        dst_idx = int(cn.attributes["to"].value)

                        # get the atom types
                        src_type = atom_idxs_to_types[src_idx]
                        dst_type = atom_idxs_to_types[dst_idx]
                        bond_src_type = atom_types_name_to_class[src_type]
                        bond_dst_type = atom_types_name_to_class[dst_type]
                        # canonicalize bond_type
                        bond_src_type, bond_dst_type, src_idx, dst_idx = dual_sort(
                            bond_src_type, bond_dst_type, src_idx, dst_idx
                        )
                        key_tuple = (bond_src_type, bond_dst_type)
                        assert key_tuple in hb_bond_set

                        if key_tuple not in bond_type_map:
                            bond_type_map[key_tuple] = len(bond_type_map)

                        bond_param_idxs.append(bond_type_map[key_tuple])

                        bond_list.append((src_idx, dst_idx))
                        bond_iota += 1

                    elif cn.nodeName == "Bond" and args.method == "template_bond":
                        # looks only at the template bond types, but this "under-symmetrizes"
                        src_idx = int(cn.attributes["from"].value)
                        dst_idx = int(cn.attributes["to"].value)

                        # get the atom types
                        src_type = atom_idxs_to_types[src_idx]
                        dst_type = atom_idxs_to_types[dst_idx]

                        # canonicalize bond_type
                        bond_src_type, bond_dst_type, src_idx, dst_idx = dual_sort(src_type, dst_type, src_idx, dst_idx)
                        bond_key = (bond_src_type, bond_dst_type)

                        if bond_key not in bond_type_map:
                            bond_type_map[bond_key] = len(bond_type_map)

                        bond_param_idxs.append(bond_type_map[bond_key])

                        bond_list.append((src_idx, dst_idx))
                        bond_iota += 1

                    elif cn.nodeName == "Bond" and args.method == "nonbonded":
                        # generate BCCs based on nonbonded atom types formed by the concatenation of q,sig,eps
                        # this roughly generates the correct symmetries (probably at the cost of having more parameters)
                        # Note: the xml is always ordered Atom tags are before Bond tags
                        src_idx = int(cn.attributes["from"].value)
                        dst_idx = int(cn.attributes["to"].value)
                        src_type = atom_idxs_to_types[src_idx]
                        dst_type = atom_idxs_to_types[dst_idx]
                        # the following also applies to method == "harmonic_bond" and method == "template_bond"
                        # it's important that we canonicalize the bond in a way that also maintains the directionality of the bcc (placeholder'd by param_idx)
                        # Note that the bond list is ordered, such that [(atom_i, atom_j), bcc] is defined such that atom_i is incremented by bcc, atom_j is decrement by bcc
                        # So, given a bond: hash_4 (atom_5) ---- hash_2 (atom_6), the convention is such that atom_6 is incremented and atom_5 is decremented
                        # to achieve this, we:
                        # 1) generate a canonical key hash formed sorting the hashes of the two atoms:  hash_2|hash_4
                        # 2) if we ended up swapping the order (as in the case above), we also need to swap the order of the atoms as well
                        # 3) record the atom idxs [(atom_6, atom_2), bcc]
                        src_hash, dst_hash, src_idx, dst_idx = dual_sort(
                            nb_atom_types_hash[src_type], nb_atom_types_hash[dst_type], src_idx, dst_idx
                        )
                        key_hash = src_hash + "|" + dst_hash
                        if key_hash not in bond_types_to_param_idx:
                            bond_types_to_param_idx[key_hash] = len(bond_types_to_param_idx)

                        bond_list.append((src_idx, dst_idx))
                        bond_param_idxs.append(bond_types_to_param_idx[key_hash])
                        bond_iota += 1

                print("atom symbols", atom_symbols)
                print("bond list", bond_list)
                print("bcc parameters", bond_param_idxs)

                # uncomment if you only want to generate types for standard residues
                # if res.attributes["name"].value.lower() in STANDARD_RESIDUES:
                all_residues.append(res_name)
                all_bond_classes.append(bond_param_idxs)
                res_mol = make_residue_mol(atom_symbols, bond_list)
                res_mol.SetProp("_Name", res.attributes["name"].value)
                all_res_mols.append(res_mol)
                all_bond_idxs.append(bond_list)

    highlightBondLists = []
    highlightBondColors = []

    # generate colors for bonds
    for bond_classes in all_bond_classes:
        bl = []
        bond_colors_complete = {}
        for idx in range(len(bond_classes)):
            bond_colors_complete[idx] = tuple(np.random.random(3))
        bond_colors = {}
        for bond_idx, bond_class in enumerate(bond_classes):
            bl.append(bond_idx)
            bond_colors[bond_idx] = bond_colors_complete[bond_class]

        highlightBondLists.append(bl)
        highlightBondColors.append(bond_colors)

    with open("res_out_" + args.method + ".svg", "w") as fh:
        svg = Draw.MolsToGridImage(
            all_res_mols,
            molsPerRow=4,
            useSVG=True,
            legends=[x.GetProp("_Name") for x in all_res_mols],
            highlightBondLists=highlightBondLists,
            highlightBondColors=highlightBondColors,
        )
        fh.write(svg)

    all_patterns_and_params = []
    for res_name, bond_idxs_list, bond_classes in zip(all_residues, all_bond_idxs, all_bond_classes):
        bc_dict: Dict[int, List] = {}
        for bc, bil in zip(bond_classes, bond_idxs_list):
            if bc not in bc_dict:
                bc_dict[bc] = []
            bc_dict[bc].append(bil)

        for v in bc_dict.values():
            pattern = res_name + " " + str(v)
            value = 0.0
            all_patterns_and_params.append([pattern, value])

    final_dict = {"EnvironmentBCC": {"patterns": all_patterns_and_params}}

    # add a special type for tip3p water assuming HOH
    final_dict["EnvironmentBCC"]["patterns"].append(["HOH [(0, 1), (0, 2)]", 0.0])

    pp = pprint.PrettyPrinter(width=500, compact=False, indent=2)
    pp.pprint(final_dict)


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert an AMBER protein XML to a timemachine FF")
    parser.add_argument("--input_path", help="Path to XML ff", required=True)
    parser.add_argument(
        "--method",
        help="Which typing system to use, either based on residue template bond tables,  harmonic bonds or the nonbonded terms",
        choices=["template_bond", "harmonic_bond", "nonbonded"],
        default="nonbonded",
    )
    args = parser.parse_args()

    process_ff(args)
