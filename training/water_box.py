import numpy as np
from simtk import unit
from simtk import openmm
from simtk.openmm import app
from simtk.openmm import Vec3

from simtk.openmm.app import PDBFile, PDBxFile

import io

def prep_system(box_width):
    # if model not in supported_models:
        # raise Exception("Specified water model '%s' is not in list of supported models: %s" % (model, str(supported_models)))

    # Load forcefield for solvent model and ions.
    # force_fields = ['tip3p.xml']
    # if ionic_strength != 0.0*unit.molar:
        # force_fields.append('amber99sb.xml')  # For the ions.
    ff = app.ForceField('tip3p.xml')

    # Create empty topology and coordinates.
    top = app.Topology()
    pos = unit.Quantity((), unit.angstroms)

    # Create new Modeller instance.
    m = app.Modeller(top, pos)

    boxSize = Vec3(box_width, box_width, box_width)*unit.nanometers
    # boxSize = unit.Quantity(numpy.ones([3]) * box_edge / box_edge.unit, box_edge.unit)
    m.addSolvent(ff, boxSize=boxSize, model='tip3p')

    system = ff.createSystem(
        m.getTopology(),
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False
    )

    positions = m.getPositions()

    positions = unit.Quantity(np.array(positions / positions.unit), positions.unit)


    # pdb_str = io.StringIO()
    fname = "debug.pdb"

    fhandle = open(fname, "w")

    PDBFile.writeHeader(m.getTopology(), fhandle)
    PDBFile.writeModel(m.getTopology(), positions, fhandle, 0)
    PDBFile.writeFooter(m.getTopology(), fhandle)

    return system, positions, np.eye(3)*box_width, fname

    assert 0

    # , positiveIon=positive_ion,
                 # negativeIon=negative_ion, ionicStrength=ionic_strength)

    # Get new topology and coordinates.
    newtop = m.getTopology()
    newpos = m.getPositions()

    # Convert positions to numpy.
    positions = unit.Quantity(numpy.array(newpos / newpos.unit), newpos.unit)

    # Create OpenMM System.
    system = ff.createSystem(
        newtop,
        nonbondedMethod=nonbondedMethod,
        nonbondedCutoff=cutoff,
        constraints=None,
        rigidWater=constrained,
        removeCMMotion=False
    )

    # Set switching function and dispersion correction.
    forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}

    forces['NonbondedForce'].setUseSwitchingFunction(False)
    if switch_width is not None:
        forces['NonbondedForce'].setUseSwitchingFunction(True)
        forces['NonbondedForce'].setSwitchingDistance(cutoff - switch_width)

    forces['NonbondedForce'].setUseDispersionCorrection(dispersion_correction)
    forces['NonbondedForce'].setEwaldErrorTolerance(ewaldErrorTolerance)

    n_atoms = system.getNumParticles()
    self.ndof = 3 * n_atoms - (constrained * n_atoms)

    self.topology = m.getTopology()
    self.system = system
    self.positions = positions
