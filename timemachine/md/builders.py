import os
from typing import Union

import numpy as np
from openmm import Vec3, app, unit

from timemachine.ff import sanitize_water_ff


def strip_units(coords):
    return np.array(coords.value_in_unit_system(unit.md_unit_system))


def build_protein_system(host_pdbfile: Union[app.PDBFile, str], protein_ff: str, water_ff: str, margin: float = 0.15):
    """
    Build a solvated protein system with a 10A padding.

    Parameters
    ---------
    host_pdbfile: str or app.PDBFile
        PDB of the host structure

    protein_ff: str
        Name of the OpenMM protein forcefield, without the .xml extension

    water_ff: str
        Name of the OpenMM water forcefield, without the .xml extension

    margin: float
        The amount of margin to add to the box. If set to 0.0 system has large forces.
        Defaults to 0.15 nanometers, empirically to reduce energies based on Hif2a

    Returns
    -------
    5-tuple of OpenMM System, coords, box, OpenMM topology and the number of waters in the system
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

    print("building a protein system with", nha, "protein atoms and", nwa, "water atoms")
    solvated_host_system = host_ff.createSystem(
        modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )

    # Add the margin to the final box
    box += np.eye(3) * margin

    return solvated_host_system, solvated_host_coords, box, modeller.topology, nwa


def build_water_system(box_width: float, water_ff: str, margin: float = 0.1):
    """
    Build a water system using OpenMM

    Parameters
    ----------
    box_width: float
        The length of each edge of the box, in nanometers

    water_ff: str
        Name of the OpenMM water forcefield, without the .xml extension

    margin: float
        The amount of margin to add to the solvent box. If set to 0.0 system has large forces.
        Defaults to 0.1 nanometers, empirically selected  to reduce energies based in 4.0nm water boxes

    Returns
    -------
    4-tuple of OpenMM System, coords, box and OpenMM topology
    """
    ff = app.ForceField(f"{water_ff}.xml")

    # Create empty topology and coordinates.
    top = app.Topology()
    pos = unit.Quantity((), unit.angstroms)
    m = app.Modeller(top, pos)

    boxSize = Vec3(box_width, box_width, box_width) * unit.nanometers
    m.addSolvent(ff, boxSize=boxSize, model=sanitize_water_ff(water_ff))

    system = ff.createSystem(m.getTopology(), nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)

    positions = m.getPositions()
    positions = strip_units(positions)

    assert m.getTopology().getNumAtoms() == positions.shape[0]

    # Construct final box with the margin
    box = np.eye(3) * (box_width + margin)

    # TODO: minimize the water box (BFGS or scipy.optimize)
    return system, positions, box, m.getTopology()
