import os
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from openmm import app, unit
from rdkit import Chem

from timemachine.fe.utils import get_romol_conf
from timemachine.ff import sanitize_water_ff
from timemachine.potentials.jax_utils import idxs_within_cutoff


def strip_units(coords) -> NDArray[np.float64]:
    return np.array(coords.value_in_unit_system(unit.md_unit_system))


def remove_clashy_waters(
    modeller: app.Modeller,
    box: NDArray[np.float64],
    water_idxs: NDArray,
    mols: List[Chem.Mol],
    clash_distance: float = 0.3,
):
    """Remove waters from an OpenMM modeler that clash with a set of molecules

    Parameters
    ----------
    modeller: app.Modeller
        Modeller to update in place

    box: NDArray[np.float64]
        Box to evaluate PBCs under

    water_idxs: NDArray[int]
        The water idxs to consider for deletion

    mols: List[Mol]
        List of molecules to determine which waters are clashy

    clash_distance: float
        Distance from a ligand atom to a water atom to consider as a clash, in nanometers
    """
    water_coords = strip_units(modeller.positions)[water_idxs]
    ligand_coords = np.concatenate([get_romol_conf(mol) for mol in mols])
    clashy_idxs = idxs_within_cutoff(water_coords, ligand_coords, box, cutoff=clash_distance)
    all_atoms = list(modeller.topology.atoms())
    waters_to_delete = set()
    for idx in clashy_idxs:
        atom = all_atoms[idx]
        waters_to_delete.add(atom.residue)
    modeller.delete(list(waters_to_delete))


def build_protein_system(
    host_pdbfile: Union[app.PDBFile, str], protein_ff: str, water_ff: str, mols: Optional[List[Chem.Mol]] = None
):
    """
    Build a solvated protein system with a 10A padding.

    Parameters
    ----------
    host_pdbfile: str or app.PDBFile
        PDB of the host structure

    protein_ff: str
        The protein forcefield name (excluding .xml) to parametrize the host_pdbfile with.

    water_ff: str
        The water forcefield name (excluding .xml) to parametrize the water with.

    mols: optional list of mols
        Molecules to be part of the system, will remove water molecules that clash with the mols.

    Returns
    -------
    5-Tuple
        OpenMM host system, coordinates, box, OpenMM topology, number of water atoms
    """

    host_ff = app.ForceField(f"{protein_ff}.xml", f"{water_ff}.xml")
    if isinstance(host_pdbfile, str):
        assert os.path.exists(host_pdbfile)
        host_pdb = app.PDBFile(host_pdbfile)
    elif isinstance(host_pdbfile, app.PDBFile):
        host_pdb = host_pdbfile
    else:
        raise TypeError("host_pdbfile must be a string or an openmm PDBFile object")

    modeller = app.Modeller(host_pdb.topology, host_pdb.positions)
    host_coords = strip_units(host_pdb.positions)

    padding = 1.0
    box_lengths = np.amax(host_coords, axis=0) - np.amin(host_coords, axis=0)

    box_lengths = box_lengths + padding
    box = np.eye(3, dtype=np.float64) * box_lengths

    modeller.addSolvent(
        host_ff, boxSize=np.diag(box) * unit.nanometers, neutralize=False, model=sanitize_water_ff(water_ff)
    )
    solvated_host_coords = strip_units(modeller.positions)

    nha = host_coords.shape[0]
    nwa = solvated_host_coords.shape[0] - nha
    if mols is not None:
        water_idxs = np.arange(nha, solvated_host_coords.shape[0])
        remove_clashy_waters(modeller, box, water_idxs, mols)
        solvated_host_coords = strip_units(modeller.positions)

    assert modeller.getTopology().getNumAtoms() == solvated_host_coords.shape[0]

    print("building a protein system with", nha, "protein atoms and", nwa, "water atoms")
    solvated_host_system = host_ff.createSystem(
        modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )

    return solvated_host_system, solvated_host_coords, box, modeller.topology, nwa


def build_water_system(box_width: float, water_ff: str, mols: Optional[List[Chem.Mol]] = None):
    """
    Build a water system with a cubic box with each side of length box_width.

    Parameters
    ---------
    box_width: float
        The length of each size of the box

    water_ff: str
        The water forcefield name (excluding .xml) to parametrize the water with.

    mols: optional list of mols
        Molecules to be part of the system, will remove water molecules that clash with the mols.

    Returns
    -------
    4-Tuple
        OpenMM host system, coordinates, box, OpenMM topology
    """
    ff = app.ForceField(f"{water_ff}.xml")

    # Create empty topology and coordinates.
    top = app.Topology()
    pos = unit.Quantity((), unit.angstroms)
    modeller = app.Modeller(top, pos)
    host_coords = strip_units(modeller.positions)

    box = np.eye(3, dtype=np.float64) * box_width
    modeller.addSolvent(ff, boxSize=np.diag(box) * unit.nanometers, model=sanitize_water_ff(water_ff))

    solvated_host_coords = strip_units(modeller.getPositions())

    if mols is not None:
        water_idxs = np.arange(host_coords.shape[0], solvated_host_coords.shape[0])
        remove_clashy_waters(modeller, box, water_idxs, mols)
        solvated_host_coords = strip_units(modeller.positions)

    assert modeller.getTopology().getNumAtoms() == solvated_host_coords.shape[0]

    system = ff.createSystem(modeller.getTopology(), nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)

    # TODO: minimize the water box (BFGS or scipy.optimize)
    return system, solvated_host_coords, np.eye(3) * box_width, modeller.getTopology()
