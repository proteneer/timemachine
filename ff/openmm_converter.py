import os
import numpy as np

from simtk import openmm as mm
from simtk.openmm import app
from simtk.openmm.app import PDBFile
from simtk.openmm.app import forcefield as ff
from simtk import unit

from ff.system import System

from timemachine import constants

def value(quantity):
    return quantity.value_in_unit_system(unit.md_unit_system)

def deserialize_system(system, cutoff=10000):
    """
    Deserialize an OpenMM XML file

    Parameters
    ----------
    filepath: str
        Location to an existing xml file to be deserialized

    Returns

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

    nrg_fns = {}


    lambda_plane_idxs = np.zeros(len(masses), dtype=np.int32)
    lambda_offset_idxs = np.zeros(len(masses), dtype=np.int32)


    for force in system.getForces():

        if isinstance(force, mm.HarmonicBondForce):
            bond_idxs = []
            param_idxs = []

            for b_idx in range(force.getNumBonds()):
                src_idx, dst_idx, length, k = force.getBondParameters(b_idx)
                length = value(length)
                k = value(k)

                k_idx = insert_parameters(k, 2)
                b_idx = insert_parameters(length, 3)

                param_idxs.append([k_idx, b_idx])
                bond_idxs.append([src_idx, dst_idx])

            bond_idxs = np.array(bond_idxs, dtype=np.int32)
            param_idxs = np.array(param_idxs, dtype=np.int32)

            nrg_fns["HarmonicBond"] = (bond_idxs, param_idxs)

        if isinstance(force, mm.HarmonicAngleForce):

            angle_idxs = []
            param_idxs = []

            for a_idx in range(force.getNumAngles()):

                src_idx, mid_idx, dst_idx, angle, k = force.getAngleParameters(a_idx)
                angle = value(angle)
                k = value(k)

                k_idx = insert_parameters(k, 0)
                a_idx = insert_parameters(angle, 1)

                param_idxs.append([k_idx, a_idx])
                angle_idxs.append([src_idx, mid_idx, dst_idx])

            angle_idxs = np.array(angle_idxs, dtype=np.int32)
            param_idxs = np.array(param_idxs, dtype=np.int32)

            nrg_fns["HarmonicAngle"] = (angle_idxs, param_idxs)

        if isinstance(force, mm.PeriodicTorsionForce):

            torsion_idxs = []
            param_idxs = []

            for t_idx in range(force.getNumTorsions()):
                a_idx, b_idx, c_idx, d_idx, period, phase, k = force.getTorsionParameters(t_idx)

                phase = value(phase)
                k = value(k)

                k_idx = insert_parameters(k, 4)
                phase_idx = insert_parameters(phase, 5)
                period_idx = insert_parameters(period, 6)

                param_idxs.append([k_idx, phase_idx, period_idx])
                torsion_idxs.append([a_idx, b_idx, c_idx, d_idx])

            torsion_idxs = np.array(torsion_idxs, dtype=np.int32)
            param_idxs = np.array(param_idxs, dtype=np.int32)
            nrg_fns["PeriodicTorsion"] = (torsion_idxs, param_idxs)

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

                charge_idx = insert_parameters(charge, 14)
                sig_idx = insert_parameters(sig, 10)
                eps_idx = insert_parameters(eps, 11)

                charge_param_idxs.append(charge_idx)
                lj_param_idxs.append([sig_idx, eps_idx])

            charge_param_idxs = np.array(charge_param_idxs, dtype=np.int32)

            print("net protein charge:", np.sum(np.array(global_params)[charge_param_idxs]))
            lj_param_idxs = np.array(lj_param_idxs, dtype=np.int32)

            # 1 here means we fully remove the interaction
            # 1-2, 1-3
            scale_idx_full = insert_parameters(1.0, 20)

            # 1-4, remove half of the interaction
            scale_idx_half = insert_parameters(0.5, 21)

            exclusion_idxs = []
            exclusion_param_idxs = []

            # fix me for scaling
            all_sig = np.array(global_params)[lj_param_idxs[:, 0]]
            all_eps = np.array(global_params)[lj_param_idxs[:, 1]]
            for a_idx in range(force.getNumExceptions()):
                src, dst, new_cp, new_sig, new_eps = force.getExceptionParameters(a_idx)
                new_sig = value(new_sig)
                new_eps = value(new_eps)

                src_sig = all_sig[src]
                dst_sig = all_sig[dst]

                src_eps = all_eps[src]
                dst_eps = all_eps[dst]
                expected_sig = (src_sig + dst_sig)/2
                expected_eps = np.sqrt(src_eps*dst_eps)

                exclusion_idxs.append([src, dst])
                if new_eps != 0:
                    np.testing.assert_almost_equal(expected_sig, new_sig)
                    np.testing.assert_almost_equal(new_eps/expected_eps, 0.5)

                    exclusion_param_idxs.append(scale_idx_half)
                else:
                    exclusion_param_idxs.append(scale_idx_full)


            exclusion_idxs = np.array(exclusion_idxs, dtype=np.int32)
            exclusion_param_idxs = np.array(exclusion_param_idxs, dtype=np.int32)

            nrg_fns["Nonbonded"] = (
                charge_param_idxs,
                lj_param_idxs,
                exclusion_idxs,
                exclusion_param_idxs,
                exclusion_param_idxs,
                lambda_plane_idxs,
                lambda_offset_idxs,
                cutoff
            )

        if isinstance(force, mm.GBSAOBCForce):

            num_atoms = force.getNumParticles()
            scale_matrix = np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)

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
                
                radius_idx = insert_parameters(radius, 12)
                scale_idx = insert_parameters(scale, 13)
                
                radius_param_idxs.append(radius_idx)
                scale_param_idxs.append(scale_idx)               

    # post-process GBSA
    nrg_fns["GBSA"] = (
        np.array(charge_param_idxs, dtype=np.int32),
        np.array(radius_param_idxs, dtype=np.int32),
        np.array(scale_param_idxs, dtype=np.int32),
        np.array(lambda_plane_idxs, dtype=np.int32),
        np.array(lambda_offset_idxs, dtype=np.int32),
        alpha,                         # alpha
        beta,                          # beta
        gamma,                         # gamma
        dielectric_offset,             # dielectric_offset
        surface_tension,               # surface_tension
        solute_dielectric,             # solute_dielectric
        solvent_dielectric,            # solvent_dieletric
        probe_radius,                  # probe_radius
        cutoff,                        # cutoff radii
        cutoff                         # cutoff force
    )

    global_params = np.array(global_params)
    global_param_groups = np.array(global_param_groups) + 100

    return System(nrg_fns, global_params, global_param_groups, np.array(masses))
