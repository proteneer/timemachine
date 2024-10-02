from collections import defaultdict
from typing import DefaultDict, List, Tuple

import numpy as np
import openmm as mm
from openmm import unit

from timemachine import constants, potentials

ORDERED_FORCES = ["HarmonicBond", "HarmonicAngle", "PeriodicTorsion", "Nonbonded"]


def value(quantity):
    return quantity.value_in_unit_system(unit.md_unit_system)


def deserialize_nonbonded_force(force, N):
    num_atoms = force.getNumParticles()

    charge_params_ = []
    lj_params_ = []

    for a_idx in range(num_atoms):
        charge, sig, eps = force.getParticleParameters(a_idx)
        charge = value(charge) * np.sqrt(constants.ONE_4PI_EPS0)

        sig = value(sig)
        eps = value(eps)

        # increment eps by 1e-3 if we have eps==0 to avoid a singularity in parameter derivatives
        # override default amber types

        # this doesn't work for water!
        # if eps == 0:
        # print("Warning: overriding eps by 1e-3 to avoid a singularity")
        # eps += 1e-3

        # charge_params.append(charge_idx)
        charge_params_.append(charge)
        lj_params_.append((sig, eps))

    charge_params = np.array(charge_params_, dtype=np.float64)

    # print("Protein net charge:", np.sum(np.array(global_params)[charge_param_idxs]))
    lj_params = np.array(lj_params_, dtype=np.float64)

    # 1 here means we fully remove the interaction
    # 1-2, 1-3
    # scale_full = insert_parameters(1.0, 20)

    # 1-4, remove half of the interaction
    # scale_half = insert_parameters(0.5, 21)

    exclusion_idxs_ = []
    scale_factors_ = []

    all_sig = lj_params[:, 0]
    all_eps = lj_params[:, 1]

    # validate exclusions/exceptions to make sure they make sense
    for a_idx in range(force.getNumExceptions()):
        # tbd charge scale factors
        src, dst, new_cp, new_sig, new_eps = force.getExceptionParameters(a_idx)
        new_q = value(new_cp) * constants.ONE_4PI_EPS0
        new_sig = value(new_sig)
        new_eps = value(new_eps)

        src_sig = all_sig[src]
        dst_sig = all_sig[dst]

        src_eps = all_eps[src]
        dst_eps = all_eps[dst]
        expected_sig = (src_sig + dst_sig) / 2
        expected_eps = np.sqrt(src_eps * dst_eps)

        src_q = charge_params[src]
        dst_q = charge_params[dst]

        expected_q = src_q * dst_q

        exclusion_idxs_.append([src, dst])

        # sanity check this (expected_eps can be zero), redo this thing

        # the lj_scale factor measures how much we *remove*
        if expected_eps == 0:
            if new_eps == 0:
                lj_scale_factor = 1
            else:
                raise RuntimeError("Divide by zero in epsilon calculation")
        else:
            lj_scale_factor = 1 - new_eps / expected_eps

        if expected_q == 0:
            if new_q == 0:
                q_scale_factor = 1
            else:
                raise RuntimeError("Divide by zero in charge calculation")
        else:
            q_scale_factor = 1 - new_q / expected_q

        scale_factors_.append((q_scale_factor, lj_scale_factor))

        # check combining rules for sigmas are consistent
        if new_eps != 0:
            np.testing.assert_almost_equal(expected_sig, new_sig)

    exclusion_idxs = np.array(exclusion_idxs_, dtype=np.int32)

    # cutoff = 1000.0

    nb_params = np.concatenate(
        [
            np.expand_dims(charge_params, axis=1),
            lj_params,
            np.zeros((N, 1)),  # 4D coordinates
        ],
        axis=1,
    )

    # optimizations
    nb_params[:, constants.NBParamIdx.LJ_SIG_IDX] = nb_params[:, 1] / 2
    nb_params[:, constants.NBParamIdx.LJ_EPS_IDX] = np.sqrt(nb_params[:, 2])

    beta = 2.0  # erfc correction

    # use the same scale factors for electrostatics and lj
    scale_factors = np.array(scale_factors_)

    return nb_params, exclusion_idxs, beta, scale_factors


def deserialize_system(system: mm.System, cutoff: float) -> Tuple[List[potentials.BoundPotential], List[float]]:
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
    a singularity from occurring in the lennard jones derivatives

    """

    masses = []

    for p in range(system.getNumParticles()):
        masses.append(value(system.getParticleMass(p)))

    N = len(masses)

    # this should not be a dict since we may have more than one instance of a given
    # force.

    bps_dict: DefaultDict[str, List[potentials.BoundPotential]] = defaultdict(list)

    for force in system.getForces():
        if isinstance(force, mm.HarmonicBondForce):
            bond_idxs_ = []
            bond_params_ = []

            for b_idx in range(force.getNumBonds()):
                src_idx, dst_idx, length, k = force.getBondParameters(b_idx)
                length = value(length)
                k = value(k)

                bond_idxs_.append([src_idx, dst_idx])
                bond_params_.append((k, length))

            bond_idxs = np.array(bond_idxs_, dtype=np.int32)
            bond_params = np.array(bond_params_, dtype=np.float64)
            bps_dict["HarmonicBond"].append(potentials.HarmonicBond(bond_idxs).bind(bond_params))

        if isinstance(force, mm.HarmonicAngleForce):
            angle_idxs_ = []
            angle_params_ = []

            for a_idx in range(force.getNumAngles()):
                src_idx, mid_idx, dst_idx, angle, k = force.getAngleParameters(a_idx)
                angle = value(angle)
                k = value(k)

                angle_idxs_.append([src_idx, mid_idx, dst_idx])
                angle_params_.append((k, angle))

            angle_idxs = np.array(angle_idxs_, dtype=np.int32)
            angle_params = np.array(angle_params_, dtype=np.float64)

            bps_dict["HarmonicAngle"].append(potentials.HarmonicAngle(angle_idxs).bind(angle_params))

        if isinstance(force, mm.PeriodicTorsionForce):
            torsion_idxs_ = []
            torsion_params_ = []

            for t_idx in range(force.getNumTorsions()):
                a_idx, b_idx, c_idx, d_idx, period, phase, k = force.getTorsionParameters(t_idx)

                phase = value(phase)
                k = value(k)

                torsion_params_.append((k, phase, period))
                torsion_idxs_.append([a_idx, b_idx, c_idx, d_idx])

            torsion_idxs = np.array(torsion_idxs_, dtype=np.int32)
            torsion_params = np.array(torsion_params_, dtype=np.float64)
            bps_dict["PeriodicTorsion"].append(potentials.PeriodicTorsion(torsion_idxs).bind(torsion_params))

        if isinstance(force, mm.NonbondedForce):
            nb_params, exclusion_idxs, beta, scale_factors = deserialize_nonbonded_force(force, N)

            bps_dict["Nonbonded"].append(
                potentials.Nonbonded(N, exclusion_idxs, scale_factors, beta, cutoff).bind(nb_params)
            )

            # nrg_fns.append(('Exclusions', (exclusion_idxs, scale_factors, es_scale_factors)))

    # ugh, ... various parts of our code assume the bps are in a certain order
    # so put them back in that order here
    bps = []
    for k in ORDERED_FORCES:
        if bps_dict.get(k):
            bps.extend(bps_dict[k])

    return bps, masses
