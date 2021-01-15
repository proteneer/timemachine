import numpy as np
from simtk import unit
from simtk.openmm import app, Vec3


def strip_units(coords):
    return unit.Quantity(np.array(coords / coords.unit), coords.unit)

def build_protein_system(host_pdbfile):

    host_ff = app.ForceField('amber99sbildn.xml', 'tip3p.xml')
    host_pdb = app.PDBFile(host_pdbfile)

    modeller = app.Modeller(host_pdb.topology, host_pdb.positions)
    host_coords = strip_units(host_pdb.positions)

    padding = 1.0
    box_lengths = np.amax(host_coords, axis=0) - np.amin(host_coords, axis=0)
    box_lengths = box_lengths.value_in_unit_system(unit.md_unit_system)

    box_lengths = box_lengths+padding
    box = np.eye(3, dtype=np.float64)*box_lengths

    modeller.addSolvent(host_ff, boxSize=np.diag(box)*unit.nanometers, neutralize=False)
    solvated_host_coords = strip_units(modeller.positions)

    nha = host_coords.shape[0]
    nwa = solvated_host_coords.shape[0] - nha

    print(nha, "protein atoms", nwa, "water atoms")
    solvated_host_system = host_ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False
    )

    return solvated_host_system, solvated_host_coords, nwa, nha, box, modeller.topology

def build_water_system(box_width):
    ff = app.ForceField('tip3p.xml')

    # Create empty topology and coordinates.
    top = app.Topology()
    pos = unit.Quantity((), unit.angstroms)
    m = app.Modeller(top, pos)

    boxSize = Vec3(box_width, box_width, box_width)*unit.nanometers
    m.addSolvent(ff, boxSize=boxSize, model='tip3p')

    system = ff.createSystem(
        m.getTopology(),
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False
    )

    positions = m.getPositions()
    positions = unit.Quantity(np.array(positions / positions.unit), positions.unit)

    assert m.getTopology().getNumAtoms() == positions.shape[0]

    # TODO: minimize the water box (BFGS or scipy.optimize)
    return system, positions, np.eye(3)*box_width, m.getTopology()
