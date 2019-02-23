import numpy as np
import ctypes
import unittest

from openeye import oechem
from openeye.oechem import OEMol, OEParseSmiles, OEAddExplicitHydrogens, OEGetIsotopicWeight, OEGetAverageWeight
from openeye import oeomega
from openeye.oechem import OEFloatArray

from timemachine import minimizer
from timemachine import system_builder


from openforcefield.utils import get_data_filename, generateTopologyFromOEMol
from openforcefield.typing.engines.smirnoff import get_molecule_parameterIDs, ForceField


def mol_coords_to_numpy_array(mol):
    coords = OEFloatArray(mol.GetMaxAtomIdx() * 3)
    mol.GetCoords(coords)
    arr = np.ctypeslib.as_array(ctypes.cast(int(coords.PtrCast()), ctypes.POINTER(ctypes.c_float)), shape=(len(coords),))
    return np.array(arr.reshape((-1, 3)))


class TestMinimizer(unittest.TestCase):

    def test_newton_cg(self):
        mol = OEMol()
        OEParseSmiles(mol, 'c1ccccc1CCCC')
        OEAddExplicitHydrogens(mol)
        # masses = get_masses(mol)
        num_atoms = mol.NumAtoms()

        omegaOpts = oeomega.OEOmegaOptions()
        omegaOpts.SetMaxConfs(1)
        omega = oeomega.OEOmega(omegaOpts)

        if not omega(mol):
            assert 0

        x0 = mol_coords_to_numpy_array(mol)/10
        ff = ForceField(get_data_filename('forcefield/smirnoff99Frosst.offxml'))
        nrgs, total_params, offsets = system_builder.construct_energies(ff, mol)
        minimizer.minimize_newton_cg(nrgs, x0, total_params)


if __name__ == "__main__":

    unittest.main()