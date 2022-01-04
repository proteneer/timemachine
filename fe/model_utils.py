import tempfile

import numpy as np

from simtk.openmm import app
from rdkit import Chem
from rdkit.Geometry import Point3D


def apply_hmr(masses, bond_list, multiplier=2):
    """
    Implements hydrogen mass repartitioning. Hydrogen masses
    are increased by multiplied by multiplier, and the connecting
    heavy atom has its mass decreased by the same amount.

    Parameters
    ----------
    masses: np.ndarray
        List of masses

    bond_list: np.ndarray, Nx2
        Nx2 array of bond pairs.

    multiplier: float
        How much to multiply the hydrogen mass by.

    Returns
    -------
    np.array
        Adjusted masses

    """

    masses = np.array(masses)  # make a copy

    def is_hydrogen(i):
        return np.abs(masses[i] - 1.00794) < 1e-3

    for i, j in bond_list:
        i, j = np.array([i, j])[np.argsort([masses[i], masses[j]])]
        if is_hydrogen(i):
            if is_hydrogen(j):
                # H-H, skip
                continue
            else:
                # H-X
                # order of operations is important!
                masses[j] -= multiplier * masses[i]
                masses[i] += multiplier * masses[i]
        else:
            # do nothing
            continue

    return masses


def image_molecule(mol_coords, box):
    """Given a set of coordinates for a single molecule and a box will return
    the coordinates wrapped into the periodic box.

    Parameters
    ----------
    mol_coords: np.ndarray
        List of coordinates that make up a molecule

    box: np.ndarray
        Periodic box, expected to be 3x3

    Returns
    -------
    np.ndarray
        Molecule coordinates imaged into box
    """
    assert box.shape == (3, 3)
    assert mol_coords.shape[1] == 3
    centroid = np.mean(mol_coords, axis=0)
    box_diag = np.diagonal(box)
    new_center = box_diag * np.floor(centroid / box_diag)
    return mol_coords - new_center


def generate_openmm_topology(objs, coords, out_filename=None, box=None):
    rd_mols = []

    mol_sizes = []
    # Convert nm into angstrom
    coords_angstroms = coords * 10
    box_angstroms = None
    if box is not None:
        box_angstroms = box * 10
    for obj in objs:
        if isinstance(obj, app.Topology):
            with tempfile.NamedTemporaryFile(mode="w") as fp:
                # write
                app.PDBFile.writeHeader(obj, fp)
                app.PDBFile.writeModel(obj, coords_angstroms[: obj.getNumAtoms()], fp, 0)
                app.PDBFile.writeFooter(obj, fp)
                fp.flush()
                romol = Chem.MolFromPDBFile(fp.name, removeHs=False)
                if romol is None:
                    raise ValueError("Failed to write pdb")
                rd_mols.append(romol)
                mol_sizes.append(obj.getNumAtoms())

        elif isinstance(obj, Chem.Mol):
            rd_mols.append(obj)
            mol_sizes.append(obj.GetNumAtoms())

        else:
            assert 0

    # exclusive prefix sum over the size of each object
    offsets = np.cumsum(mol_sizes) - mol_sizes
    combined_mol = None

    for mol_idx, mol in enumerate(rd_mols):

        mol_copy = Chem.Mol(mol)
        mol_copy.RemoveAllConformers()
        mol_conf = Chem.Conformer(mol.GetNumAtoms())

        start_idx = offsets[mol_idx]
        if mol_idx == len(offsets) - 1:
            mol_pos = coords_angstroms[start_idx:]
        else:
            end_idx = offsets[mol_idx + 1]
            mol_pos = coords_angstroms[start_idx:end_idx]
        if box is not None:
            mol_pos = image_molecule(mol_pos, box_angstroms)

        if mol_pos.shape[0] != mol.GetNumAtoms():
            raise ValueError(f"Coordinates shape don't match {mol_pos.shape[0]} != {mol.GetNumAtoms()}")

        for a_idx, pos in enumerate(mol_pos):
            mol_conf.SetAtomPosition(a_idx, pos.astype(np.float64))
        mol_copy.AddConformer(mol_conf)
        if combined_mol is None:
            combined_mol = mol_copy
        else:
            combined_mol = Chem.CombineMols(combined_mol, mol_copy)

    if out_filename is None:
        fp = tempfile.NamedTemporaryFile(mode="w")
        out_filename = fp.name

    Chem.MolToPDBFile(combined_mol, out_filename)
    combined_pdb = app.PDBFile(out_filename)
    return combined_pdb.topology
