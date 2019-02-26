import time
import numpy as np
import unittest
import ctypes

from openeye import oechem
from openeye.oechem import OEMol, OEParseSmiles, OEAddExplicitHydrogens, OEGetIsotopicWeight, OEGetAverageWeight
from openeye import oeomega
from openeye.oechem import OEFloatArray

from openforcefield.utils import get_data_filename, generateTopologyFromOEMol
from openforcefield.typing.engines.smirnoff import get_molecule_parameterIDs, ForceField

from timemachine import minimizer
from timemachine.constants import BOLTZ
from timemachine import system_builder
from timemachine.cpu_functionals import custom_ops

from simtk import openmm

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

def write_xyz(ofs, mol, coords):
    mol.SetCoords(OEFloatArray(coords.reshape(-1)))
    oechem.OEWriteMolecule(ofs, mol)

class TestSmallMolecule(unittest.TestCase):

    def test_mol(self):

        mol = OEMol()
        # OEParseSmiles(mol, 'CCOCCSCC')
        # OEParseSmiles(mol, 'c1ccccc1')
        # OEParseSmiles(mol, 'C1CCCCC1O')
        OEParseSmiles(mol, 'C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O')
        OEAddExplicitHydrogens(mol)
        masses = get_masses(mol)
        num_atoms = mol.NumAtoms()


        topology = generateTopologyFromOEMol(mol)


        # ff = ForceField(get_data_filename('forcefield/Frosst_AlkEthOH.offxml') )
        ff = ForceField(get_data_filename('forcefield/smirnoff99Frosst.offxml') )
        # labels = ff.labelMolecules( [mol], verbose = True )

        omm_system = ff.createSystem(topology, [mol])

        print(type(omm_system))
        with open("system.xml", "w") as fh:
            fh.write(openmm.openmm.XmlSerializer.serialize(omm_system))

        omegaOpts = oeomega.OEOmegaOptions()
        omegaOpts.SetMaxConfs(800)
        omega = oeomega.OEOmega(omegaOpts)
        omega.SetStrictStereo(False)
        omega.SetSampleHydrogens(True)
        omega.SetEnergyWindow(15.0)
        omega.SetRMSThreshold(1.0)

        if not omega(mol):
            assert 0

        nrgs, total_params, offsets = system_builder.construct_energies(ff, mol)
        # dt = 0.0025
        # friction = 10.0
        # temperature = 300

        # gradient descent
        dt = 0.001
        friction = 10.0
        temperature = 10

        a,b,c = get_abc_coefficents(masses, dt, friction, temperature)

        buf_size = estimate_buffer_size(1e-16, a)
        x0 = mol_coords_to_numpy_array(mol)/10

        x0 = minimizer.minimize_newton_cg(nrgs, x0, total_params)

        intg = custom_ops.Integrator_double(
            dt,
            buf_size,
            num_atoms,
            total_params,
            a,
            b,
            c
        )


        context = custom_ops.Context_double(
            nrgs,
            intg
        )

        # it's very important that we start with equilibrium geometries
        # x0 = np.array([[ 0.06672798, -0.08789801,  0.17259836],
        #      [ 0.16416019, -0.00393655,  0.25996411],
        #      [ 0.22437823,  0.07365441,  0.1361862],
        #      [ 0.13155428,  0.19842917,  0.15364743],
        #      [ 0.00140648, -0.02683543,  0.10997669],
        #      [ 0.12003344, -0.15992539,  0.11009348],
        #      [ 0.0019258,  -0.14449902,  0.23957494],
        #      [ 0.2228789,  -0.09036981,  0.29247978],
        #      [ 0.23281977,  0.05531402,  0.32119114],
        #      [ 0.31699611,  0.12723394,  0.15912026],
        #      [ 0.28942673, -0.01347244,  0.12423653],
        #      [ 0.19188221,  0.28887075,  0.16359857],
        #      [ 0.07279447,  0.20940621,  0.06246885],
        #      [ 0.06117841,  0.19766579,  0.23744114]])


        origin = np.sum(x0, axis=0)/x0.shape[0]


        print("X0", x0)
        print(np.sum(np.power(x0[0,: ] - x0[1,:], 2)))
        # assert 0

        for atom in mol.GetAtoms():
            print(atom)

        num_steps = 1000000

        ofs = oechem.oemolostream("new_frames.xyz")
        ofs.SetFormat(oechem.OEFormat_XYZ)

        # for i in range(10):
        #     write_xyz(ofs, mol, x0)
        # assert 0
        intg.set_coordinates(x0.reshape(-1).tolist())
        intg.set_velocities(np.zeros_like(x0).reshape(-1).tolist())
        start_time = time.time()
        for step in range(num_steps):

            if step % 1000 == 0:
                # print(step)
                coords = np.array(intg.get_coordinates()).reshape((-1, 3))

                center = np.sum(coords, axis=0)/coords.shape[0]

                dto = np.sqrt(np.sum(np.power(center - origin, 2)))
                velocities = np.array(intg.get_velocities()).reshape((-1, 3))

                print(coords, velocities)
                net_mass = np.sum(masses)
                nv = np.sum(np.expand_dims(np.array(masses),axis=-1)*velocities, axis=0)
                cc_bond_length = np.sqrt(np.sum(np.power(coords[0,: ] - coords[1,:], 2)))

                write_xyz(ofs, mol, np.array(coords)*10)

                dxdp = np.array(intg.get_dxdp()).reshape((total_params, num_atoms, 3))
                amax, amin = np.amax(dxdp), np.amin(dxdp)
                print(step, "\tdto\t", dto, "\tnv\t", nv, "\tcc_bond_length\t", cc_bond_length, "\tamax/amin", amax, "\t", amin)
                print("OFFSETS", offsets)

                segments = np.split(dxdp, offsets)[1:]
                for grads, force in zip(segments, nrgs):
                    # print(force)
                    if isinstance(force, custom_ops.HarmonicBondGPU_double):
                        grads = grads.reshape((-1, 2, num_atoms, 3))
                        print("Bond Constants:", np.amax(grads[:, 0, :, :]), np.amin(grads[:, 0, :, :]))
                        print("Bond Lengths:", np.amax(grads[:, 1, :, :]), np.amin(grads[:, 1, :, :]))
                        # print(grads[:, 1, :, :])
                    elif isinstance(force, custom_ops.HarmonicAngleGPU_double):
                        grads = grads.reshape((-1, 2, num_atoms, 3))
                        print("Angle Constants:", np.amax(grads[:, 0, :, :]), np.amin(grads[:, 0, :, :]))
                        # print(grads[:, 0, :, :])
                        print("Angle Radians:", np.amax(grads[:, 1, :, :]), np.amin(grads[:, 1, :, :]))
                        # print(grads[:, 1, :, :])
                    elif isinstance(force, custom_ops.PeriodicTorsionGPU_double):
                        grads = grads.reshape((-1, 3, num_atoms, 3))
                        print("Torsion Constants:", np.amax(grads[:, 0, :, :]), np.amin(grads[:, 0, :, :]))
                        print("Torsion Phase:",  np.amax(grads[:, 1, :, :]), np.amin(grads[:, 1, :, :]))
                        print("Torsion Periods:",  np.amax(grads[:, 2, :, :]), np.amin(grads[:, 2, :, :]))
                    elif isinstance(force, custom_ops.LennardJonesGPU_double):
                        grads = grads.reshape((-1, 2, num_atoms, 3))
                        print("LJ sigma:", np.amax(grads[:, 0, :, :]), np.amin(grads[:, 0, :, :]))
                        print("LJ epsilon:",  np.amax(grads[:, 1, :, :]), np.amin(grads[:, 1, :, :]))
                    elif isinstance(force, custom_ops.ElectrostaticsGPU_double):
                        grads = grads.reshape((-1, num_atoms, num_atoms, 3))
                        print("CHARGE amax/amin", np.amax(grads), np.amin(grads))
                        # print("LJ sigma:", np.amax(grads[:, 0, :, :]), np.amin(grads[:, 0, :, :]))
                        # print("LJ epsilon:",  np.amax(grads[:, 1, :, :]), np.amin(grads[:, 1, :, :]))
                    else:
                        assert 0
                if np.any(np.isnan(dxdp)):
                    assert 0
            context.step()

        print(x0)
        assert 0

        # simulation is stable but we get NaNs?
        # assert 0

        print("time per step:", (time.time() - start_time)/num_steps)
        print(offsets)
        print("total number of parameters:", total_params)
        # looks pretty stable
        dxdp = np.array(intg.get_dxdp()).reshape((total_params, num_atoms, 3))

        segments = np.split(dxdp, offsets)[1:]
        for grads, force in zip(segments, nrgs):
            print(force)
            if isinstance(force, custom_ops.HarmonicBondGPU_float):
                grads = grads.reshape((-1, 2, num_atoms, 3))
                print("Bond Constants:")
                print(grads[:, 0, :, :])
                print("Bond Lengths:")
                print(grads[:, 1, :, :])
            elif isinstance(force, custom_ops.HarmonicAngleGPU_float):
                grads = grads.reshape((-1, 2, num_atoms, 3))
                print("Angle Constants:")
                print(grads[:, 0, :, :])
                print("Angle Radians:")
                print(grads[:, 1, :, :])
            elif isinstance(force, custom_ops.PeriodicTorsionGPU_float):
                grads = grads.reshape((-1, 3, num_atoms, 3))
                print("Torsion Constants:")
                print(grads[:, 0, :, :])
                print("Torsion Phase:")
                print(grads[:, 1, :, :])
                print("Torsion Periods:")
                print(grads[:, 2, :, :])
            elif isinstance(force, custom_ops.LennardJonesGPU_float):
                grads = grads.reshape((-1, 2, num_atoms, 3))
                print("LJ sigma:")
                print(grads[:, 0, :, :])
                print("LJ epsilon:")
                print(grads[:, 1, :, :])

        # visualize the trjajectory

        # print(intg.)

if __name__ == "__main__":
    unittest.main()
