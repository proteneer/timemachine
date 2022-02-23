from rdkit import Chem
from rdkit.Chem import Draw


def rgb_to_decimal(x, y, z):
    return x / 255, y / 255, z / 255


def draw_mol(mol, core):
    highlightAtoms = [x for x in core]
    return Draw.DrawMolecule(mol, highlightAtoms=highlightAtoms)


def draw_dummy_core_ixns(mol, core, bonds, dummy_group, color_blind=False):
    """
    Draw a grid of molecules with interactions between atoms in dummy_group
    and the core highlighted.
    """

    if color_blind:
        COLOR_DUMMY_IXN = rgb_to_decimal(230, 159, 0)
        COLOR_DUMMY_ACTIVE = rgb_to_decimal(240, 228, 66)
        COLOR_DUMMY_INACTIVE = rgb_to_decimal(0, 158, 115)
        COLOR_CORE_ACTIVE = rgb_to_decimal(213, 94, 0)
        COLOR_CORE_INACTIVE = rgb_to_decimal(204, 121, 167)
        COLOR_BOND = (0.96, 0.74, 0)
    else:
        COLOR_DUMMY_IXN = (0, 0.7, 0)
        COLOR_DUMMY_ACTIVE = (0.6, 1, 0.6)
        COLOR_DUMMY_INACTIVE = (0.188, 0.835, 0.784)
        COLOR_CORE_ACTIVE = (0.9, 0.5, 0.5)
        COLOR_CORE_INACTIVE = (1, 0.8, 0.8)
        COLOR_BOND = (0.92, 0.1, 0.95)

    assert len(set(core).intersection(set(dummy_group))) == 0

    highlightAtomLists = []
    highlightBondLists = []
    highlightAtomColorsLists = []
    highlightBondColorsLists = []
    all_mols = []

    bonds.sort(key=len)

    legends = []

    for atom_idxs in bonds:

        mol_copy = Chem.Mol(mol)
        highlightAtomColors = {}
        highlightBondColors = {}
        highlightAtoms = set()
        highlightBonds = set()

        # default colors
        for a in range(mol_copy.GetNumAtoms()):
            highlightAtoms.add(a)
            if a in core:
                highlightAtomColors[a] = COLOR_CORE_INACTIVE
            elif a in dummy_group:
                highlightAtomColors[a] = COLOR_DUMMY_ACTIVE
            else:
                highlightAtomColors[a] = COLOR_DUMMY_INACTIVE

        # interacting atoms
        for a in atom_idxs:
            atom = mol_copy.GetAtomWithIdx(int(a))
            atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
            if a in core:
                highlightAtomColors[a] = COLOR_CORE_ACTIVE
            else:
                highlightAtomColors[a] = COLOR_DUMMY_IXN

        is_improper = False
        for idx in range(len(atom_idxs)):
            if idx != len(atom_idxs) - 1:
                bond = mol_copy.GetBondBetweenAtoms(int(atom_idxs[idx]), int(atom_idxs[idx + 1]))

                # this may be none if we have an improper torsion
                if bond is None:
                    if len(atom_idxs) == 4:
                        # improper
                        is_improper = True
                    else:
                        assert 0, "Bad idxs"

                else:
                    highlightBonds.add(bond.GetIdx())
                    highlightBondColors[bond.GetIdx()] = COLOR_BOND

        highlightAtomLists.append(list(range(mol_copy.GetNumAtoms())))
        highlightBondLists.append(list(highlightBonds))
        highlightAtomColorsLists.append(highlightAtomColors)
        highlightBondColorsLists.append(highlightBondColors)
        if is_improper:
            label = "improper"
        elif len(atom_idxs) == 2:
            label = "bond"
        elif len(atom_idxs) == 3:
            label = "angle"
        elif len(atom_idxs) == 4:
            label = "proper"
        else:
            assert 0

        all_mols.append(mol_copy)
        legends.append(label + " " + repr([int(x) for x in atom_idxs]))

    return Draw.MolsToGridImage(
        all_mols,
        molsPerRow=4,
        highlightAtomLists=highlightAtomLists,
        highlightAtomColors=highlightAtomColorsLists,
        highlightBondLists=highlightBondLists,
        highlightBondColors=highlightBondColorsLists,
        subImgSize=(250, 250),
        legends=legends,
        useSVG=True,
    )
