import numpy as np
import simtk.unit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D


def to_md_units(q):
    return q.value_in_unit_system(simtk.unit.md_unit_system)


def write(xyz, masses, recenter=True):
    if recenter:
        xyz = xyz - np.mean(xyz, axis=0, keepdims=True)
    buf = str(len(masses)) + "\n"
    buf += "timemachine\n"
    for m, (x, y, z) in zip(masses, xyz):
        if int(round(m)) == 12:
            symbol = "C"
        elif int(round(m)) == 14:
            symbol = "N"
        elif int(round(m)) == 16:
            symbol = "O"
        elif int(round(m)) == 32:
            symbol = "S"
        elif int(round(m)) == 35:
            symbol = "Cl"
        elif int(round(m)) == 1:
            symbol = "H"
        elif int(round(m)) == 31:
            symbol = "P"
        elif int(round(m)) == 19:
            symbol = "F"
        elif int(round(m)) == 80:
            symbol = "Br"
        elif int(round(m)) == 127:
            symbol = "I"
        else:
            raise Exception("Unknown mass:" + str(m))

        buf += symbol + " " + str(round(x, 5)) + " " + str(round(y, 5)) + " " + str(round(z, 5)) + "\n"
    return buf


def convert_uIC50_to_kJ_per_mole(amount_in_uM):
    """
    TODO: more sig figs
    """
    return 0.593 * np.log(amount_in_uM * 1e-6) * 4.18


def convert_uM_to_kJ_per_mole(amount_in_uM):
    """
    Convert a potency measurement in uM concentrations.

    Parameters
    ----------
    amount_in_uM: float
        Binding potency in uM concentration.

    Returns
    -------
    float
        Binding potency in kJ/mol.

    """
    return 0.593 * np.log(amount_in_uM * 1e-6) * 4.18


# TODO: add a module for atom-mapping, with RDKit MCS based and other approaches

# TODO: add a visualization module?
# TODO: compare with perses atom map visualizations?


def draw_mol(mol, highlightAtoms, highlightColors):
    """from YTZ, Feb 1, 2021"""
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 200)
    drawer.DrawMolecule(mol, highlightAtoms=highlightAtoms, highlightAtomColors=highlightColors)
    drawer.FinishDrawing()

    # TODO: return or save image, for inclusion in a PDF report or similar

    # To display in a notebook:
    # svg = drawer.GetDrawingText().replace('svg:', '')
    # display(SVG(svg))


def plot_atom_mapping(mol_a, mol_b, core):
    """from YTZ, Feb 1, 2021

    TODO: move this into a SingleTopology.visualize() or SingleTopology.debug() method"""
    print(repr(core))
    atom_colors_a = {}
    atom_colors_b = {}
    for (a_idx, b_idx), rgb in zip(core, np.random.random((len(core), 3))):
        atom_colors_a[int(a_idx)] = tuple(rgb.tolist())
        atom_colors_b[int(b_idx)] = tuple(rgb.tolist())

    draw_mol(mol_a, core[:, 0].tolist(), atom_colors_a)
    draw_mol(mol_b, core[:, 1].tolist(), atom_colors_b)


def plot_atom_mapping_grid(mol_a, mol_b, core, show_idxs=False):
    mol_a_2d = Chem.Mol(mol_a)
    mol_b_2d = Chem.Mol(mol_b)

    AllChem.Compute2DCoords(mol_a_2d)
    AllChem.GenerateDepictionMatching2DStructure(mol_b_2d, mol_a_2d, atomMap=core.tolist())

    atom_colors_a = {}
    atom_colors_b = {}
    for (a_idx, b_idx), rgb in zip(core, np.random.random((len(core), 3))):
        atom_colors_a[int(a_idx)] = tuple(rgb.tolist())
        atom_colors_b[int(b_idx)] = tuple(rgb.tolist())

    if show_idxs:
        for atom in mol_a_2d.GetAtoms():
            atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
        for atom in mol_b_2d.GetAtoms():
            atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))

    return Draw.MolsToGridImage(
        [mol_a_2d, mol_b_2d],
        molsPerRow=2,
        highlightAtomLists=[core[:, 0].tolist(), core[:, 1].tolist()],
        highlightAtomColors=[atom_colors_a, atom_colors_b],
        subImgSize=(400, 400),
        legends=[mol_a.GetProp("_Name"), mol_b.GetProp("_Name")],
        useSVG=True,
    )


def get_romol_bonds(mol):
    """
    Return bond idxs given a mol. These are not canonicalized.
    """
    bond_list = []
    for bond in mol.GetBonds():
        bond_list.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return bond_list


def get_romol_conf(mol):
    """Coordinates of mol's 0th conformer, in nanometers"""
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf / 10  # from angstroms to nm


def set_romol_conf(mol, new_coords):
    """Sets coordinates of mol's 0th conformer"""
    assert new_coords.shape[0] == mol.GetNumAtoms()
    # convert from nm to angstroms
    angstrom_coords = new_coords * 10
    conf = mol.GetConformer(0)
    for i, (x, y, z) in enumerate(angstrom_coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))


def get_mol_masses(mol):
    """Return the masses for the given mol"""
    return np.array([a.GetMass() for a in mol.GetAtoms()])


def get_mol_name(mol) -> str:
    """Return the title for the given mol"""
    return mol.GetProp("_Name")


def sanitize_energies(full_us, lamb_idx, cutoff=10000):
    """
    Given a matrix with F rows and K columns,
    we sanitize entries that differ by more than cutoff.

    That is, given full_us:
    [
        [15000.0, -5081923.0, 1598, 1.5, -23.0],
        [-423581.0, np.nan, -238, 13.5,  23.0]
    ]
    And lamb_idx 3 and cutoff of 10000,
    full_us is sanitized to:

    [
        [inf, inf, 1598, 1.5, -23.0],
        [inf, inf, -238, 13.5,  23.0]
    ]

    Parameters
    ----------
    full_us: NDArray of shape (F, K)
        Matrix of full energies

    lamb_idx: int
        Which of the K windows to serve as the reference energy

    cutoff: float
        Used to determine the threshold for a "good" energy

    Returns
    -------
    np.array of shape (F,K)
        Sanitized energies

    """
    ref_us = np.expand_dims(full_us[:, lamb_idx], axis=1)
    abs_us = np.abs(full_us - ref_us)
    return np.where(abs_us < cutoff, full_us, np.inf)


def extract_delta_Us_from_U_knk(U_knk):
    """
    Generate delta_Us from the U_knk matrix for use with BAR.

    Parameters
    ----------
    U_knk: NDArray of shape (K, N, K)
        Energies matrix, K simulations ran with N frames with
        energies evaluated at K states

    Returns
    -------
    np.array of shape (K-1, 2, N)
        Returns the delta_Us of the fwd and rev processes

    """

    assert U_knk.shape[0] == U_knk.shape[-1]

    K = U_knk.shape[0]

    def delta_U(from_idx, to_idx):
        """
        Computes [U(x, to_idx) - U(x, from_idx) for x in xs]
        where xs are simulated at from_idx
        """
        current = U_knk[from_idx]
        current_energies = current[:, from_idx]
        perturbed_energies = current[:, to_idx]
        return perturbed_energies - current_energies

    delta_Us = []

    for lambda_idx in range(K - 1):
        # lambda_us have shape (F, K)
        fwd_delta_U = delta_U(lambda_idx, lambda_idx + 1)
        rev_delta_U = delta_U(lambda_idx + 1, lambda_idx)
        delta_Us.append((fwd_delta_U, rev_delta_U))

    return np.array(delta_Us)
