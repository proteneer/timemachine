import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom

import unittest
from fe import atom_mapping

class TestAtomMapping(unittest.TestCase):

    # def test_map_a_to_b(self):
    #     a = Chem.MolFromSmiles("IC1=C(I)C(F)1")
    #     b = Chem.MolFromSmiles("FC1C(Br)=C(Br)1")
    #     a_to_b = atom_mapping.mcs_map(a, b)
    #     np.testing.assert_equal(a_to_b, {
    #         5: 0,
    #         4: 1,
    #         1: 2,
    #         2: 4
    #     })

    def test_minimal_distance_mapping(self):
        a = Chem.MolFromSmiles("C(F)(F)(F)I")
        b = Chem.MolFromSmiles("C(F)(Br)(F)F")
        a_to_b = atom_mapping.mcs_map(a, b)
