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

    # def test_openmm(self):



    def test_mol(self):

        mol = OEMol()
        # OEParseSmiles(mol, 'CCOCCSCC')
        OEParseSmiles(mol, 'CCC')
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
        # assert 0




        nrgs, total_params, offsets = system_builder.construct_energies(ff, mol)

        dt = 0.001
        friction = 10.0
        temperature = 100

        # gradient descent
        # dt = 0.001
        # friction = 100000.0
        # temperature = 0

        a,b,c = get_abc_coefficents(masses, dt, friction, temperature)

        buf_size = estimate_buffer_size(1e-16, a)
        # print("BUFFER SIZE", buf_size)



        omegaOpts = oeomega.OEOmegaOptions()
        omegaOpts.SetMaxConfs(1)
        omega = oeomega.OEOmega(omegaOpts)

        if not omega(mol):
            assert 0
        x0 = mol_coords_to_numpy_array(mol)/10


        # DEBUG
        # test_sys = openmm.openmm.XmlSerializer.deserialize(open("debug_force.xml").read())
        # timestep = 1.0
        # integrator = openmm.VerletIntegrator(timestep)

        # platform = openmm.Platform.getPlatformByName('Reference')
        # context = openmm.Context(test_sys, integrator, platform)
        # # state = openmm.openmm.State()
        # # print(dir(state))
        # context.setPositions(x0)
        # state = context.getState(getEnergy=True, getForces=True)
        # print(state.getForces(asNumpy=True))
        # print(state.getPotentialEnergy())


        # nrg_, grad_, hess_, mps_ = nrgs[-1].total_derivative(x0, 1000)
        # print(grad_)
        # print(nrg_)

        # assert 0


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

        # x0 = np.array([ [ 8.05472061e-02, -5.44342399e-02,  4.95790727e-02],
        #                 [ 2.16504470e-01, -4.92023071e-03, -1.60400651e-03],
        #                 [ 1.00858870e-03,  1.18017523e-03, -1.03775455e-04],
        #                 [ 6.62027746e-02, -1.61187708e-01,  3.03640384e-02],
        #                 [ 7.40195438e-02, -3.69146653e-02,  1.56997517e-01],
        #                 [ 2.97981262e-01, -5.77684566e-02,  4.83387597e-02],
        #                 [ 2.26062834e-01,  1.01577170e-01,  1.98065490e-02],
        #                 [ 2.24628925e-01, -1.94601193e-02, -1.09533176e-01]], dtype=np.float32)


        # x0 = np.array([[ 0.04047876 ,-0.07297815,  0.07083304],
        # [ 0.2542038  , 0.16208108,  0.29189807],
        # [ 0.16372064 ,-0.01820499,  0.14100419],
        # [ 0.13159984 , 0.10884641,  0.21945369],
        # [-0.0018707  , 0.00117548,  0.00234415],
        # [ 0.06478791 ,-0.16292682,  0.01338776],
        # [-0.03638209 ,-0.09927345,  0.1441983 ],
        # [ 0.29465136 , 0.08722229,  0.36077148],
        # [ 0.22823274 , 0.25117514,  0.3499414 ],
        # [ 0.33288103 , 0.18954685,  0.22090748],
        # [ 0.20357928 ,-0.09469703,  0.2086626 ],
        # [ 0.24129161 , 0.00267996,  0.06640348],
        # [ 0.09366605 , 0.1860415 ,  0.1514599 ],
        # [ 0.05258387 , 0.08845842,  0.29265663]])

        origin = np.sum(x0, axis=0)/x0.shape[0]


        print("X0", x0)
        print(np.sum(np.power(x0[0,: ] - x0[1,:], 2)))
        # assert 0

        for atom in mol.GetAtoms():
            print(atom)

        num_steps = 100000

        ofs = oechem.oemolostream("new_frames.xyz")
        ofs.SetFormat(oechem.OEFormat_XYZ)

        # for i in range(10):
        #     write_xyz(ofs, mol, x0)
        # assert 0

        intg.set_coordinates(x0.reshape(-1).tolist())
        intg.set_velocities(np.zeros_like(x0).reshape(-1).tolist())

        start_time = time.time()

        # wriet out XYZ
        # assert 0


        for step in range(num_steps):

            if step % 500 == 0:
                # print(step)
                coords = np.array(intg.get_coordinates()).reshape((-1, 3))
                print(coords)
                center = np.sum(coords, axis=0)/coords.shape[0]

                # void ReferenceRemoveCMMotionKernel::execute(ContextImpl& context) {
                #     if (data.stepCount%frequency != 0)
                #         return;
                #     vector<Vec3>& velData = extractVelocities(context);
                    
                #     // Calculate the center of mass momentum.
                    
                #     double momentum[] = {0.0, 0.0, 0.0};
                #     double mass = 0.0;
                #     for (size_t i = 0; i < masses.size(); ++i) {
                #         momentum[0] += masses[i]*velData[i][0];
                #         momentum[1] += masses[i]*velData[i][1];
                #         momentum[2] += masses[i]*velData[i][2];
                #         mass += masses[i];
                #     }
                    
                #     // Adjust the particle velocities.
                    
                #     momentum[0] /= mass;
                #     momentum[1] /= mass;
                #     momentum[2] /= mass;
                #     for (size_t i = 0; i < masses.size(); ++i) {
                #         if (masses[i] != 0.0) {
                #             velData[i][0] -= momentum[0];
                #             velData[i][1] -= momentum[1];
                #             velData[i][2] -= momentum[2];
                #         }
                #     }
                # }


                dto = np.sqrt(np.sum(np.power(center - origin, 2)))
                velocities = np.array(intg.get_velocities()).reshape((-1, 3))
                net_mass = np.sum(masses)
                # nv = np.sum(np.expand_dims(np.array(masses), axis=-1)*velocities, axis=0)
                nv = np.sum(np.expand_dims(np.array(masses),axis=-1)*velocities, axis=0)
                # assert 0
                cc_bond_length = np.sqrt(np.sum(np.power(coords[0,: ] - coords[1,:], 2)))

                write_xyz(ofs, mol, np.array(coords)*10)

                dxdp = np.array(intg.get_dxdp()).reshape((total_params, num_atoms, 3))
                amax, amin = np.amax(dxdp), np.amin(dxdp)
                print(step, "\tdto\t", dto, "\tnv\t", nv, "\tcc_bond_length\t", cc_bond_length, "\tamax/amin", amax, "\t", amin)

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
