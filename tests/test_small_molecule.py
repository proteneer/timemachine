import time
import numpy as np
import unittest
import ctypes

from openeye.oechem import OEMol, OEParseSmiles, OEAddExplicitHydrogens, OEGetIsotopicWeight, OEGetAverageWeight
from openeye import oeomega
from openeye.oechem import OEFloatArray

from openforcefield.utils import get_data_filename
from openforcefield.typing.engines.smirnoff import get_molecule_parameterIDs, ForceField

from timemachine.constants import BOLTZ
from timemachine import system_builder
from timemachine.cpu_functionals import custom_ops


def get_abc_coefficents(
    masses,
    dt,
    friction,
    temperature):
    """
    Get a,b,c coefficients of the integrator.
    """
    vscale = np.exp(-dt*friction)
    if friction == 0:
        fscale = dt
    else:
        fscale = (1-vscale)/friction

    kT = BOLTZ * temperature
    nscale = np.sqrt(kT*(1-vscale*vscale)) # noise scale
    invMasses = (1.0/masses).reshape((-1, 1))
    sqrtInvMasses = np.sqrt(invMasses)

    coeff_a = vscale
    coeff_bs = fscale*invMasses
    coeff_cs = nscale*sqrtInvMasses

    return coeff_a, coeff_bs, coeff_cs

def estimate_buffer_size(epsilon, coeff_a):
    """
    Estimate optimal size of the buffer to achieve convergence
    """
    return np.int64(np.log(epsilon)/np.log(coeff_a)+1)


def get_masses(mol):
    masses = []
    for atom in mol.GetAtoms():
        elem = atom.GetAtomicNum()
        mass = atom.GetIsotope()
        masses.append(OEGetIsotopicWeight(elem, mass))

    return np.array(masses)

def mol_coords_to_numpy_array(mol):
    coords = OEFloatArray(mol.GetMaxAtomIdx() * 3)
    mol.GetCoords(coords)
    arr = np.ctypeslib.as_array(ctypes.cast(int(coords.PtrCast()), ctypes.POINTER(ctypes.c_float)), shape=(len(coords),))
    return np.array(arr.reshape((-1, 3)))

class TestSmallMolecule(unittest.TestCase):

    def test_mol(self):
        mol = OEMol()
        OEParseSmiles(mol, 'C1CCCC1CCCCC')
        OEAddExplicitHydrogens(mol)
        masses = get_masses(mol)
        num_atoms = mol.NumAtoms()

        ff = ForceField(get_data_filename('forcefield/Frosst_AlkEthOH.offxml') )
        labels = ff.labelMolecules( [mol], verbose = True )

        nrgs, total_params = system_builder.construct_energies(ff, mol)

        dt = 0.0025
        friction = 10.0
        temperature = 300

        a,b,c = get_abc_coefficents(masses, dt, friction, temperature)

        buf_size = estimate_buffer_size(1e-6, a)

        intg = custom_ops.Integrator_float(
            dt,
            buf_size,
            num_atoms,
            total_params,
            a,
            b,
            c
        )

        context = custom_ops.Context_float(
            nrgs,
            intg
        )

        omegaOpts = oeomega.OEOmegaOptions()
        omegaOpts.SetMaxConfs(1)
        omega = oeomega.OEOmega(omegaOpts)

        if not omega(mol):
            assert 0

        x0 = mol_coords_to_numpy_array(mol)

        num_steps = 1000

        intg.set_coordinates(x0.reshape(-1).tolist())
        intg.set_velocities(np.zeros_like(x0).reshape(-1).tolist())

        start_time = time.time()
        for step in range(num_steps):
            context.step()
        print("time per step:", (time.time() - start_time)/num_steps)

        # looks pretty stable
        intg.get_dxdp()

        # visualize

        # print(intg.)

if __name__ == "__main__":
    unittest.main()
