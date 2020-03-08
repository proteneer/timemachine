import os
import unittest
import ast
from ff import forcefield
from ff import system
from ff import openmm_converter
import itertools
import numpy as np

import pathlib
from rdkit import Chem


from simtk.openmm import app
from simtk.openmm.app import PDBFile

def get_masses(m):
    masses = []
    for a in m.GetAtoms():
        masses.append(a.GetMass())
    return masses


class TestForcefield(unittest.TestCase):

    def get_smirnoff(self):
        cwd = pathlib.Path(__file__).parent.parent.absolute()
        fpath = os.path.join(cwd, 'ff', "smirnoff_1.1.0.py")
        ff = forcefield.Forcefield(fpath)
        return ff

    def test_forcefield(self):
        cwd = pathlib.Path(__file__).parent.parent.absolute()
        fpath = os.path.join(cwd, 'ff', "smirnoff_1.1.0.py")
        ff = forcefield.Forcefield(fpath)
        ff_res = ff.serialize()
        ff_raw = ast.literal_eval(open(fpath).read())
        self.assertDictEqual(ff_res, ff_raw)

    def test_params_and_groups(self):

        itercount = itertools.count()

        handle = {
            'Angle': {'params': [
                ['[*:1]~[#6X4:2]-[*:3]', next(itercount)/5, next(itercount)/5],
                ['[#1:1]-[#6X4:2]-[#1:3]', next(itercount)/5,next(itercount)/5],
                ['[*;r3:1]1~;@[*;r3:2]~;@[*;r3:3]1', next(itercount)/5, next(itercount)/5],
            ]},
            'Bond': {'params': [['[#6X4:1]-[#6X4:2]', next(itercount)/5, next(itercount)/5],
                      ['[#6X4:1]-[#6X3:2]', next(itercount)/5, next(itercount)/5],
                      ['[#6X4:1]-[#6X3:2]=[#8X1+0]', next(itercount)/5, next(itercount)/5],
                ]},
               'Improper': {'params': [['[*:1]~[#6X3:2](~[*:3])~[*:4]', next(itercount)/5, next(itercount)/5, next(itercount)/5],
                         ['[*:1]~[#6X3:2](~[#8X1:3])~[#8:4]', next(itercount)/5, next(itercount)/5, next(itercount)/5]
                ]},
               'Proper': {'params': [['[*:1]-[#6X4:2]-[#6X4:3]-[*:4]', [[next(itercount)/5, next(itercount)/5, next(itercount)/5]]],
                       ['[#6X4:1]-[#6X4:2]-[#6X4:3]-[#6X4:4]', [[next(itercount)/5, next(itercount)/5, next(itercount)/5], [next(itercount)/5, next(itercount)/5, next(itercount)/5], [next(itercount)/5, next(itercount)/5, next(itercount)/5]]]
                       ]},
                'vdW': {'params': [['[#1:1]', next(itercount)/5, next(itercount)/5],
                    ['[#1:1]-[#6X4]', next(itercount)/5, next(itercount)/5],
                    ],
                     'props': {'combining_rules': 'Lorentz-Berthelot', 'method': 'cutoff', 'potential': 'Lennard-Jones-12-6', 'scale12': 0.0, 'scale13': 0.0, 'scale14': 0.5, 'scale15': 1.0}
              },
              'GBSA': {
                'params': [
                  ['[*:1]', next(itercount)/5, next(itercount)/5],
                ],
               'props': {
                  'solvent_dielectric' : 78.3, # matches OBC2,
                  'solute_dielectric' : 1.0,
                  'probe_radius' : 0.14,
                  'surface_tension' : 28.3919551,
                  'dielectric_offset' : 0.009,
                  # GBOBC1
                  'alpha' : 0.8,
                  'beta' : 0.0,
                  'gamma' : 2.909125
                }
              },
              'SimpleCharges': {
                'params': [
                    ['[#1:1]', next(itercount)/5],
                    ['[#1:1]-[#6X4]', next(itercount)/5],
                ]
              }
            }
        
        ff = forcefield.Forcefield(handle)
        # the -1 is for exclusions
        np.testing.assert_equal(np.arange(next(itercount))/5, ff.params[:-1])
        ref_groups = [0,1, 0,1, 0,1, 2,3, 2,3, 2,3, 4,5,6, 4,5,6, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 10,11, 10,11, 12,13, 14,14]
        np.testing.assert_equal(ref_groups, ff.param_groups[:-1])

    def test_parameterization(self):

        ff = self.get_smirnoff()
        mol1 = Chem.AddHs(Chem.MolFromSmiles("O=C(N)C[C@H](N)C(=O)O"))
        nrg_fns_1 = ff.parameterize(mol1)

        # mol2 = Chem.AddHs(Chem.MolFromSmiles("O=C(C)Oc1ccccc1C(=O)O"))
        # nrg_fns_2 = ff.parameterize(mol2)

    def test_merging(self):

        ff = self.get_smirnoff()

        mol1 = Chem.AddHs(Chem.MolFromSmiles("O=C(N)C[C@H](N)C(=O)O"))
        nrg_fns1 = ff.parameterize(mol1)

        mol1_masses = get_masses(mol1)

        mol2 = Chem.AddHs(Chem.MolFromSmiles("O=C(C)Oc1ccccc1C(=O)O"))
        nrg_fns2 = ff.parameterize(mol2)

        mol2_masses = get_masses(mol2)

        system1 = system.System(nrg_fns1, ff.params, ff.param_groups, mol1_masses)
        system2 = system.System(nrg_fns2, ff.params, ff.param_groups, mol2_masses)

        system3 = system1.merge(system2)

        np.testing.assert_equal(system3.masses, np.concatenate([system1.masses, system2.masses]))
        np.testing.assert_equal(system3.params, np.concatenate([system1.params, system2.params]))
        np.testing.assert_equal(system3.param_groups, np.concatenate([system1.param_groups, system2.param_groups]))

        system3.make_gradients(3, np.float64)

    def test_merging_with_openmm(self):

        ff = self.get_smirnoff()

        mol1 = Chem.AddHs(Chem.MolFromSmiles("O=C(N)C[C@H](N)C(=O)O"))
        nrg_fns1 = ff.parameterize(mol1)
        mol1_masses = get_masses(mol1)

        host_pdb = app.PDBFile("examples/BRD4_minimized.pdb")
        amber_ff = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')

        protein_system = amber_ff.createSystem(
            host_pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            rigidWater=False
        )

        system1 = system.System(nrg_fns1, ff.params, ff.param_groups, mol1_masses)
        protein_sys = openmm_converter.deserialize_system(protein_system)

        combined_system = protein_sys.merge(system1)


        np.testing.assert_equal(combined_system.masses, np.concatenate([protein_sys.masses, system1.masses]))
        np.testing.assert_equal(combined_system.params, np.concatenate([protein_sys.params, system1.params]))
        np.testing.assert_equal(combined_system.param_groups, np.concatenate([protein_sys.param_groups, system1.param_groups]))


        # open("examples/BRD4_minimized.pdb")
        # print(system)