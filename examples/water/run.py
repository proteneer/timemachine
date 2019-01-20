#!/usr/bin/env python

from simtk.openmm.app import *
from simtk.openmm import *
import simtk.unit as u

pdb = PDBFile('input.pdb')

forcefield = ForceField('tip3p.xml')

system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=Ewald,
    nonbondedCutoff=0.85*u.nanometer,
    # constraints=HBonds,
    rigidWater=False,
)

integrator = LangevinIntegrator(300*u.kelvin, 1.0/u.picosecond, 2.0*u.femtosecond)
barostat = MonteCarloBarostat(1.0*u.bar, 300.0*u.kelvin, 25)
system.addForce(barostat)

platform = Platform.getPlatformByName('Reference')
simulation = Simulation(pdb.topology, system, integrator, platform)

simulation.context.setPositions(pdb.positions)

print('Minimizing...')
simulation.minimizeEnergy()

simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
print('Equilibrating...')
# simulation.step(100000)
simulation.step(30)
assert 0

state = simulation.context.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True, getParameters=True, getParameterDerivatives=True, enforcePeriodicBox=True)
with open('state.xml', 'w') as f: f.write(XmlSerializer.serialize(state))
with open('system.xml', 'w') as f: f.write(XmlSerializer.serialize(system))

