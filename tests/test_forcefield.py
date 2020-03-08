import os
import unittest
import ast
from ff import forcefield
import itertools
import numpy as np

import pathlib


class TestForcefield(unittest.TestCase):

    def test_forcefield(self):

        cwd = pathlib.Path(__file__).parent.parent.absolute()
        fpath = os.path.join(cwd, 'ff', "smirnoff_1.1.0.py")
        ff_raw = ast.literal_eval(open(fpath).read())
        ff = forcefield.Forcefield(fpath)
        ff_res = ff.serialize()

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
        np.testing.assert_equal(np.arange(next(itercount))/5, ff.params)
        ref_groups = [0,1, 0,1, 0,1, 2,3, 2,3, 2,3, 4,5,6, 4,5,6, 7,8,9, 7,8,9, 7,8,9, 7,8,9, 10,11, 10,11, 12,13, 14,14]
        np.testing.assert_equal(ref_groups, ff.param_groups)
