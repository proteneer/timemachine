import os
from typing import Union

import numpy as np
from openmm import Vec3, app, unit

from timemachine.ff import sanitize_water_ff


def strip_units(coords):
    return unit.Quantity(np.array(coords / coords.unit), coords.unit)


def build_protein_system(host_pdbfile: Union[app.PDBFile, str], protein_ff: str, water_ff: str):
    """
    Build a solvated protein system with a 10A padding.

    Parameters
    ---------
    host_pdbfile: str or app.PDBFile
        PDB of the host structure

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
    box_lengths = box_lengths.value_in_unit_system(unit.md_unit_system)

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

    return solvated_host_system, solvated_host_coords, box, modeller.topology


def build_water_system(box_width, water_ff: str):
    ff = app.ForceField(f"{water_ff}.xml")

    # Create empty topology and coordinates.
    top = app.Topology()
    pos = unit.Quantity((), unit.angstroms)
    m = app.Modeller(top, pos)

    boxSize = Vec3(box_width, box_width, box_width) * unit.nanometers
    m.addSolvent(ff, boxSize=boxSize, model=sanitize_water_ff(water_ff))

    system = ff.createSystem(m.getTopology(), nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)

    positions = m.getPositions()
    positions = unit.Quantity(np.array(positions / positions.unit), positions.unit)

    assert m.getTopology().getNumAtoms() == positions.shape[0]

    # TODO: minimize the water box (BFGS or scipy.optimize)
    return system, positions, np.eye(3) * box_width, m.getTopology()
