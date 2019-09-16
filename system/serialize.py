import os
import numpy as np

from timemachine.lib import custom_ops

from simtk import openmm as mm
from simtk.openmm import app
from simtk.openmm.app import PDBFile
from simtk.openmm.app import forcefield as ff
from simtk import unit

def value(quantity):
    return quantity.value_in_unit_system(unit.md_unit_system)

def deserialize_system(system):
    """
    Deserialize an OpenMM XML file

    Parameters
    ----------
    filepath: str
        Location to an existing xml file to be deserialized

    """

    # filename, file_extension = os.path.splitext(filepath)
    # sys_xml = open(filepath, 'r').read()
    # system = mm.XmlSerializer.deserialize(sys_xml)
    # coords = np.loadtxt(filename + '.xyz').astype(np.float64)
    # coords = coords # WARNING DEPENDENT ON IF WE USE OPENMM OR NOT WTF

    # pdb = PDBFile('/home/yutong/structures/anon/capped_receptor.pdb')
    # forcefield = ff.ForceField('amber96.xml', 'tip3p.xml')
    # system = forcefield.createSystem(
    #     pdb.topology,
    #     nonbondedMethod=app.CutoffNonPeriodic,
    #     nonbondedCutoff=1*unit.nanometer,
    #     # constraints=HBonds
    # )
    # coords = []
    # for x, y, z in pdb.getPositions():
    #     coords.append([value(x), value(y), value(z)])
    # coords = np.array(coords)

    global_params = []
    global_param_groups = []
    test_potentials = []

    def insert_parameters(obj, group):
        """
        Attempts to insert a value v and return the index it belongs to
        """
        p_idx = len(global_params)
        global_params.append(obj)
        global_param_groups.append(group)
        return p_idx

    masses = []

    for p in range(system.getNumParticles()):
        masses.append(value(system.getParticleMass(p)))

    # print(len(masses), coords.shape[0])
    # assert len(masses) == coords.shape[0]

    for force in system.getForces():
        # print("PROCESSING", force)
        if isinstance(force, mm.HarmonicBondForce):
            bond_idxs = []
            param_idxs = []

            for b_idx in range(force.getNumBonds()):
                src_idx, dst_idx, length, k = force.getBondParameters(b_idx)
                length = value(length)


                # k = value(k)/5
                k = value(k)/16

                # print("bond K", k)

                k_idx = insert_parameters(k, 0)
                b_idx = insert_parameters(length, 1)

                param_idxs.append([k_idx, b_idx])
                bond_idxs.append([src_idx, dst_idx])

            bond_idxs = np.array(bond_idxs, dtype=np.int32)
            param_idxs = np.array(param_idxs, dtype=np.int32)

            test_hb = (
                custom_ops.HarmonicBond_f64,
                (
                    bond_idxs,
                    param_idxs
                )
            )

            test_potentials.append(test_hb)


        if isinstance(force, mm.HarmonicAngleForce):
            angle_idxs = []
            param_idxs = []

            for a_idx in range(force.getNumAngles()):

                src_idx, mid_idx, dst_idx, angle, k = force.getAngleParameters(a_idx)
                angle = value(angle)
                k = value(k)/16

                # print("ANGLE k", k)

                k_idx = insert_parameters(k, 2)
                a_idx = insert_parameters(angle, 3)

                param_idxs.append([k_idx, a_idx])
                angle_idxs.append([src_idx, mid_idx, dst_idx])

            angle_idxs = np.array(angle_idxs, dtype=np.int32)
            param_idxs = np.array(param_idxs, dtype=np.int32)

            test_ha = (custom_ops.HarmonicAngle_f64,
                (
                    angle_idxs,
                    param_idxs
                )
            )

            test_potentials.append(test_ha)

        if isinstance(force, mm.PeriodicTorsionForce):
            torsion_idxs = []
            param_idxs = []

            for t_idx in range(force.getNumTorsions()):
                a_idx, b_idx, c_idx, d_idx, period, phase, k = force.getTorsionParameters(t_idx)

                # period is unitless
                phase = value(phase)
                k = value(k)

                k_idx = insert_parameters(k, 4)
                phase_idx = insert_parameters(phase, 5)
                period_idx = insert_parameters(period, 6)

                param_idxs.append([k_idx, phase_idx, period_idx])
                torsion_idxs.append([a_idx, b_idx, c_idx, d_idx])

            torsion_idxs = np.array(torsion_idxs, dtype=np.int32)
            param_idxs = np.array(param_idxs, dtype=np.int32)

            test_ha = (custom_ops.PeriodicTorsion_f64,
                (
                    torsion_idxs,
                    param_idxs
                )
            )

            test_potentials.append(test_ha)

        if isinstance(force, mm.NonbondedForce):

            num_atoms = force.getNumParticles()
            scale_matrix = np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)

            charge_param_idxs = []
            lj_param_idxs = []

            for a_idx in range(num_atoms):
                scale_matrix[a_idx][a_idx] = 0
                charge, sig, eps = force.getParticleParameters(a_idx)

                # print('charge, sig, eps', charge, sig, eps)

                charge = value(charge)
                # print("inserting charge", charge)
                sig = value(sig)
                eps = value(eps)*10
                # if sig == 0 or eps == 0:
                    # print("WARNING: invalid sig eps detected", sig, eps, "adjusting to 0.5 and 0.0")
                    # assert eps == 0.0
                    # sig = 0.5
                    # eps = 0.0

                charge_idx = insert_parameters(charge, 7)
                sig_idx = insert_parameters(sig, 8)
                eps_idx = insert_parameters(eps, 9)

                charge_param_idxs.append(charge_idx)
                lj_param_idxs.append([sig_idx, eps_idx])

            for a_idx in range(force.getNumExceptions()):
                src, dst, _, _, _ = force.getExceptionParameters(a_idx)
                scale_matrix[src][dst] = 0
                scale_matrix[dst][src] = 0

            charge_param_idxs = np.array(charge_param_idxs, dtype=np.int32)
            lj_param_idxs = np.array(lj_param_idxs, dtype=np.int32)

            print("SCALE MATRIX", scale_matrix)

            # test_lj = (custom_ops.LennardJones_f64,
            #     (
            #         scale_matrix,
            #         lj_param_idxs
            #     )
            # )

            # test_potentials.append(test_lj)

            # test_es = (custom_ops.Electrostatics_f64,
            #     (
            #         scale_matrix,
            #         charge_param_idxs,
            #     )
            # )

            # test_potentials.append(test_es)

            # print("PROTEIN NET CHARGE", np.sum(np.array(global_params)[charge_param_idxs]))

    global_params = np.array(global_params)
    global_param_groups = np.array(global_param_groups)

    return test_potentials, (global_params, global_param_groups), np.array(masses)
