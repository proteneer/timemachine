import numpy as np
from simtk.openmm import app
import tempfile
from rdkit import Chem
import mdtraj 
from rdkit.Geometry import Point3D

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

    masses = np.array(masses) # make a copy

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


# def generate_openmm_topology(objs, host_coords, out_filename):
def generate_openmm_topology(objs, coords, out_filename=None):
    rd_mols = []

    mol_sizes = []
    coords = coords*10
    for obj in objs:
        if isinstance(obj, app.Topology):
            with tempfile.NamedTemporaryFile(mode='w') as fp:
                # write
                app.PDBFile.writeHeader(obj, fp)
                app.PDBFile.writeModel(obj, coords[:obj.getNumAtoms()], fp, 0)
                app.PDBFile.writeFooter(obj, fp)
                fp.flush()
                romol = Chem.MolFromPDBFile(fp.name, removeHs=False)
                assert romol is not None
                rd_mols.append(romol)
                mol_sizes.append(obj.getNumAtoms())

        elif isinstance(obj, Chem.Mol):
            rd_mols.append(obj)
            mol_sizes.append(obj.GetNumAtoms())
        
        else:
            assert 0

    # exclusive prefix sum over the size of each object
    offsets = np.cumsum(mol_sizes) - mol_sizes
    combined_mol = rd_mols[0]

    for mol_idx, mol in enumerate(rd_mols[1:]):
        mol_idx += 1

        mol_copy = Chem.Mol(mol)
        mol_copy.RemoveAllConformers()
        mol_conf = Chem.Conformer(mol.GetNumAtoms())

        start_idx = offsets[mol_idx]
        if mol_idx == len(offsets) - 1:
            mol_pos = coords[start_idx:]
        else:
            end_idx = offsets[mol_idx + 1]
            mol_pos = coords[start_idx:end_idx]

        assert mol_pos.shape[0] == mol.GetNumAtoms()

        for a_idx, pos in enumerate(mol_pos):
            mol_conf.SetAtomPosition(a_idx, pos.astype(np.float64))
        mol_copy.AddConformer(mol_conf)
        combined_mol = Chem.CombineMols(combined_mol, mol_copy)

    if out_filename is None:
        fp = tempfile.NamedTemporaryFile(mode='w')
        out_filename = fp.name
    
    Chem.MolToPDBFile(combined_mol, out_filename)       
    combined_pdb = app.PDBFile(out_filename)
    return combined_pdb.topology

def generate_imaged_topology(objs, x0, box0, name):
    combined_topology_dummy = generate_openmm_topology(
        objs,
        x0,
        out_filename=None
    )
    traj_dummy = mdtraj.Trajectory(np.array([x0]), mdtraj.Topology.from_openmm(combined_topology_dummy))
    traj_dummy.unitcell_vectors = np.array([box0])
    traj_dummy.image_molecules()

    combined_topology = generate_openmm_topology(
        objs,
        traj_dummy.xyz[0],
        out_filename=name
    )
    return combined_topology