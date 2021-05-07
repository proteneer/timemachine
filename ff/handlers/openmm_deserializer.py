import os
import numpy as np

from simtk import openmm as mm
from simtk.openmm import app
from simtk.openmm.app import PDBFile
from simtk.openmm.app import forcefield as ff
from simtk import unit

from timemachine import constants
from timemachine.lib import potentials

def value(quantity):
    return quantity.value_in_unit_system(unit.md_unit_system)

def deserialize_system(
    system,
    cutoff):
    """
    Deserialize an OpenMM XML file

    Parameters
    ----------
    system: openmm.System
        A system object to be deserialized
    cutoff: float
        Nonbonded cutoff, in nm

    Returns
    -------
    list of lib.Potential, masses

    Note: We add a small epsilon (1e-3) to all zero eps values to prevent
    a singularity from occuring in the lennard jones derivatives

    """

    masses = []

    for p in range(system.getNumParticles()):
        masses.append(value(system.getParticleMass(p)))

    N = len(masses)

    # this should not be a dict since we may have more than one instance of a given
    # force.
    bps = []

    for force in system.getForces():

        if isinstance(force, mm.HarmonicBondForce):
            bond_idxs = []
            bond_params = []

            for b_idx in range(force.getNumBonds()):
                src_idx, dst_idx, length, k = force.getBondParameters(b_idx)
                length = value(length)
                k = value(k)

                bond_idxs.append([src_idx, dst_idx])
                bond_params.append((k, length))

            bond_idxs = np.array(bond_idxs, dtype=np.int32)
            bond_params = np.array(bond_params, dtype=np.float64)
            bps.append(potentials.HarmonicBond(bond_idxs).bind(bond_params))

        if isinstance(force, mm.HarmonicAngleForce):

            angle_idxs = []
            angle_params = []

            for a_idx in range(force.getNumAngles()):

                src_idx, mid_idx, dst_idx, angle, k = force.getAngleParameters(a_idx)
                angle = value(angle)
                k = value(k)

                angle_idxs.append([src_idx, mid_idx, dst_idx])
                angle_params.append((k, angle))

            angle_idxs = np.array(angle_idxs, dtype=np.int32)
            angle_params = np.array(angle_params, dtype=np.float64)

            bps.append(potentials.HarmonicAngle(angle_idxs).bind(angle_params))

        if isinstance(force, mm.PeriodicTorsionForce):

            torsion_idxs = []
            torsion_params = []

            for t_idx in range(force.getNumTorsions()):
                a_idx, b_idx, c_idx, d_idx, period, phase, k = force.getTorsionParameters(t_idx)

                phase = value(phase)
                k = value(k)

                torsion_params.append((k, phase, period))
                torsion_idxs.append([a_idx, b_idx, c_idx, d_idx])

            torsion_idxs = np.array(torsion_idxs, dtype=np.int32)
            torsion_params = np.array(torsion_params, dtype=np.float64)
            bps.append(potentials.PeriodicTorsion(torsion_idxs).bind(torsion_params))

        if isinstance(force, mm.NonbondedForce):

            num_atoms = force.getNumParticles()

            charge_params = []
            lj_params = []

            for a_idx in range(num_atoms):

                charge, sig, eps = force.getParticleParameters(a_idx)
                charge = value(charge)*np.sqrt(constants.ONE_4PI_EPS0)

                sig = value(sig)
                eps = value(eps)

                # increment eps by 1e-3 if we have eps==0 to avoid a singularity in parameter derivatives
                # override default amber types

                # this doesn't work for water!
                # if eps == 0:
                    # print("Warning: overriding eps by 1e-3 to avoid a singularity")
                    # eps += 1e-3

                # charge_params.append(charge_idx)
                charge_params.append(charge)
                lj_params.append((sig, eps))

            charge_params = np.array(charge_params, dtype=np.float64)

            # print("Protein net charge:", np.sum(np.array(global_params)[charge_param_idxs]))
            lj_params = np.array(lj_params, dtype=np.float64)

            # 1 here means we fully remove the interaction
            # 1-2, 1-3
            # scale_full = insert_parameters(1.0, 20)

            # 1-4, remove half of the interaction
            # scale_half = insert_parameters(0.5, 21)

            exclusion_idxs = []
            scale_factors = []

            all_sig = lj_params[:, 0]
            all_eps = lj_params[:, 1]

            # validate exclusions/exceptions to make sure they make sense
            for a_idx in range(force.getNumExceptions()):

                # tbd charge scale factors
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

                # sanity check this (expected_eps can be zero), redo this thing

                # the lj_scale factor measures how much we *remove*
                if expected_eps == 0:
                    if new_eps == 0:
                        lj_scale_factor = 1
                    else:
                        raise RuntimeError("Divide by zero in epsilon calculation")
                else:
                    lj_scale_factor = 1 - new_eps/expected_eps

                scale_factors.append(lj_scale_factor)

                # tbd fix charge_scale_factors using new_cp
                if new_eps != 0:
                    np.testing.assert_almost_equal(expected_sig, new_sig)

            exclusion_idxs = np.array(exclusion_idxs, dtype=np.int32)

            lambda_plane_idxs = np.zeros(N, dtype=np.int32)
            lambda_offset_idxs = np.zeros(N, dtype=np.int32)

            # cutoff = 1000.0

            nb_params = np.concatenate([
                np.expand_dims(charge_params, axis=1),
                lj_params
            ], axis=1)

            # optimizations
            nb_params[:, 1] = nb_params[:, 1]/2
            nb_params[:, 2] = np.sqrt(nb_params[:, 2])

            beta = 2.0 # erfc correction

            # use the same scale factors for electrostatics and lj
            scale_factors = np.stack([
                scale_factors,
                scale_factors
            ], axis=1)


            bps.append(potentials.Nonbonded(
                exclusion_idxs,
                scale_factors,
                lambda_plane_idxs,
                lambda_offset_idxs,
                beta,
                cutoff).bind(nb_params)
            )

            # nrg_fns.append(('Exclusions', (exclusion_idxs, scale_factors, es_scale_factors)))

    return bps, masses