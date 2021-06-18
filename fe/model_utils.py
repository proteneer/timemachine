import numpy as np
from simtk.openmm import app
import tempfile
from rdkit import Chem


def apply_hmr(masses, bond_list, multiplier=2):
    """
    Implements hydrogen mass repartioning. Hydrogen masses
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
                masses[j] -= multiplier*masses[i]
                masses[i] += multiplier*masses[i]
        else:
            # do nothing
            continue

    return masses


def generate_topology(objs, host_coords, out_filename):
    rd_mols = []
    # super jank
    for obj in objs:
        if isinstance(obj, app.Topology):
            with tempfile.NamedTemporaryFile(mode='w') as fp:
                # write
                app.PDBFile.writeHeader(obj, fp)
                app.PDBFile.writeModel(obj, host_coords, fp, 0)
                app.PDBFile.writeFooter(obj, fp)
                fp.flush()
                romol = Chem.MolFromPDBFile(fp.name, removeHs=False)
                rd_mols.append(romol)

        if isinstance(obj, Chem.Mol):
            rd_mols.append(obj)

    combined_mol = rd_mols[0]
    for mol in rd_mols[1:]:
        combined_mol = Chem.CombineMols(combined_mol, mol)

    # with tempfile.NamedTemporaryFile(mode='w') as fp:
    fp = open(out_filename, "w")
    # write
    Chem.MolToPDBFile(combined_mol, out_filename)
    fp.flush()
    # read
    combined_pdb = app.PDBFile(out_filename)
    return combined_pdb.topology