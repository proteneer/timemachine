from timemachine.potentials import bonded
import numpy as onp
import jax
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
#from sys import stdout
import unittest
#import functools


class TestBonded(unittest.TestCase):
    def test_total_bonded(self):
        pdb = PDBFile('formaldehyde.pdb')
        forcefield = ForceField('formaldehyde.xml', 'amber14/tip3p.xml')
        system = forcefield.createSystem(pdb.topology)
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        context = Context(system, integrator)
        context.setPositions(pdb.positions)
        state = context.getState(getForces=True, getEnergy=True)
        openmm_forces = onp.array(state.getForces().value_in_unit(kilojoules_per_mole/nanometer))
        conf = onp.array([[1.1,-0.3,0.0],
                          [2.2,0.2,0.0],
                          [0.2,0.4,0.0],
                          [0.9,-1.4,0.0]])
        params = onp.array([47697.6,1.229,28451.2,1.090])
        bond_idxs = onp.array([[0,1],
                    [0,2],
                    [0,3]])
        param_idxs = onp.array([[0,1],
                     [2,3],
                     [2,3]])
        dEdx_fn = jax.grad(bonded.harmonic_bond, argnums=(0,))
        timemachine_bondforces = dEdx_fn(conf,params,None,bond_idxs,param_idxs)[0]
        angleparams = onp.array([2657.3,2.0943985])
        angle_idxs = onp.array([[1,0,2],[1,0,3],[2,0,3]])
        angleparam_idxs = onp.array([[0,1],[0,1],[0,1]])
        dEdx_angles = jax.grad(bonded.harmonic_angle, argnums=(0,))
        timemachine_angleforces = dEdx_angles(conf, angleparams, None, angle_idxs, angleparam_idxs, False)[0]
        torsion_params = onp.array([46.024,3.1,2])
        torsion_idxs = onp.array([[1,0,2,3]])
        torsionparam_idxs = onp.array([[0,1,2]])
        dEdx_torsions = jax.grad(bonded.periodic_torsion, argnums=(0,))
        timemachine_torsionforces = dEdx_torsions(conf, torsion_params, None, torsion_idxs, torsionparam_idxs)[0]
        #print(openmm_forces)
        #print(timemachine_angleforces)
        timemachine_total = timemachine_bondforces + timemachine_angleforces + timemachine_torsionforces
        #print(timemachine_total)
        #print(onp.linalg.norm(openmm_forces + timemachine_total))
        for i in range(4):
            for j in range(3):
                if openmm_forces[i][j] + timemachine_total[i][j] > 0.01:
                    assert False
        assert True

    #def test_total_nonbonded(self):
        #assert False



