import os
import numpy as np

from timemachine.lib import ops

from simtk import openmm as mm
from simtk.openmm import app
from simtk.openmm.app import PDBFile
from simtk.openmm.app import forcefield as ff
from simtk import unit

from timemachine import constants

def value(quantity):
    return quantity.value_in_unit_system(unit.md_unit_system)

def deserialize_system(
    system,
    dimension):
    """
    Deserialize an OpenMM XML file

    Parameters
    ----------
    filepath: str
        Location to an existing xml file to be deserialized

    """
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

    for force in system.getForces():
        if isinstance(force, mm.HarmonicBondForce):
            bond_idxs = []
            param_idxs = []

            for b_idx in range(force.getNumBonds()):
                src_idx, dst_idx, length, k = force.getBondParameters(b_idx)
                length = value(length)
                k = value(k)

                k_idx = insert_parameters(k, 0)
                b_idx = insert_parameters(length, 1)

                param_idxs.append([k_idx, b_idx])
                bond_idxs.append([src_idx, dst_idx])

            bond_idxs = np.array(bond_idxs, dtype=np.int32)
            param_idxs = np.array(param_idxs, dtype=np.int32)

            test_hb = (
                ops.HarmonicBond,
                (
                    bond_idxs,
                    param_idxs,
                    dimension
                )
            )

            test_potentials.append(test_hb)


        if isinstance(force, mm.HarmonicAngleForce):

            angle_idxs = []
            param_idxs = []

            for a_idx in range(force.getNumAngles()):

                src_idx, mid_idx, dst_idx, angle, k = force.getAngleParameters(a_idx)
                angle = value(angle)
                k = value(k)

                k_idx = insert_parameters(k, 2)
                a_idx = insert_parameters(angle, 3)

                param_idxs.append([k_idx, a_idx])
                angle_idxs.append([src_idx, mid_idx, dst_idx])

            angle_idxs = np.array(angle_idxs, dtype=np.int32)
            param_idxs = np.array(param_idxs, dtype=np.int32)

            test_ha = (ops.HarmonicAngle,
                (
                    angle_idxs,
                    param_idxs,
                    dimension
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

            test_tors = (ops.PeriodicTorsion,
                (
                    torsion_idxs,
                    param_idxs,
                    dimension
                )
            )

            test_potentials.append(test_tors)

        if isinstance(force, mm.NonbondedForce):

            num_atoms = force.getNumParticles()
            scale_matrix = np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)

            charge_param_idxs = []
            lj_param_idxs = []

            for a_idx in range(num_atoms):
                scale_matrix[a_idx][a_idx] = 0
                charge, sig, eps = force.getParticleParameters(a_idx)

                # this needs to be scaled by sqrt(eps0)
                charge = value(charge)*np.sqrt(constants.ONE_4PI_EPS0)
                sig = value(sig)
                eps = value(eps)

                charge_idx = insert_parameters(charge, 7)
                sig_idx = insert_parameters(sig, 8)
                eps_idx = insert_parameters(eps, 9)

                charge_param_idxs.append(charge_idx)
                lj_param_idxs.append([sig_idx, eps_idx])

            charge_param_idxs = np.array(charge_param_idxs, dtype=np.int32)
            lj_param_idxs = np.array(lj_param_idxs, dtype=np.int32)

            # 1 here means we fully remove the interaction
            scale_idx = insert_parameters(1.0, 10)

            exclusion_idxs = []
            exclusion_param_idxs = []

            # fix me for scaling
            for a_idx in range(force.getNumExceptions()):
                src, dst, new_cp, new_sig, new_eps = force.getExceptionParameters(a_idx)
                exclusion_idxs.append([src, dst])
                exclusion_param_idxs.append(scale_idx)

            exclusion_idxs = np.array(exclusion_idxs, dtype=np.int32)
            exclusion_param_idxs = np.array(exclusion_param_idxs, dtype=np.int32)

            test_nonbonded = (
                ops.Nonbonded,
                (
                    charge_param_idxs,
                    lj_param_idxs,
                    exclusion_idxs,
                    exclusion_param_idxs,
                    exclusion_param_idxs,
                    0.85,
                    dimension
                )
            )

            test_potentials.append(test_nonbonded)

        if isinstance(force, mm.GBSAOBCForce):

            num_atoms = force.getNumParticles()
            scale_matrix = np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)

            charge_param_idxs = []
            radius_param_idxs = []
            scale_param_idxs = []
            
            solvent_dielectric = force.getSolventDielectric()
            solute_dielectric = force.getSoluteDielectric()
            probe_radius = 0.14
            surface_tension = 28.3919551
            dielectric_offset = 0.009

            # GBOBC1
            alpha = 0.8
            beta = 0.0
            gamma = 2.909125
            
            for a_idx in range(num_atoms):

                scale_matrix[a_idx][a_idx] = 0
                charge, radius, scale = force.getParticleParameters(a_idx)

                # this needs to be scaled by sqrt(eps0)
                charge = value(charge)*np.sqrt(constants.ONE_4PI_EPS0)
                radius = value(radius)
                
                charge_idx = insert_parameters(charge, 7)
                radius_idx = insert_parameters(radius, 11)
                scale_idx = insert_parameters(scale, 12)
                
                charge_param_idxs.append(charge_idx)
                radius_param_idxs.append(radius_idx)
                scale_param_idxs.append(scale_idx)               
            
            test_gbsa = (
                ops.GBSA,
                (
                    np.array(charge_param_idxs),
                    np.array(radius_param_idxs),
                    np.array(scale_param_idxs),
                    alpha,                         # alpha
                    beta,                          # beta
                    gamma,                         # gamma
                    dielectric_offset,             # dielectric_offset
                    surface_tension,               # surface_tension
                    solute_dielectric,             # solute_dielectric
                    solvent_dielectric,            # solvent_dieletric
                    probe_radius,                  # probe_radius
                    9999,                          # cutoff
                    dimension
                )
            )

            test_potentials.append(test_gbsa)

    global_params = np.array(global_params)
    global_param_groups = np.array(global_param_groups)

    return test_potentials, (global_params, global_param_groups), np.array(masses)
