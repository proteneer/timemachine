from collections import defaultdict
from typing import DefaultDict, List, Tuple

import numpy as np
import openmm as mm
from openmm import unit

from timemachine import constants, potentials

ORDERED_FORCES = ["HarmonicBond", "HarmonicAngle", "PeriodicTorsion", "Nonbonded"]


def value(quantity):
    return quantity.value_in_unit_system(unit.md_unit_system)


def deserialize_system(
    system: mm.System, cutoff: float, verbose=False
) -> Tuple[List[potentials.BoundPotential], List[float]]:
    """
    Deserialize an OpenMM XML file

    Parameters
    ----------
    system: openmm.System
        A system object to be deserialized
    cutoff: float
        Nonbonded cutoff, in nm
    verbose: bool
        print warnings when overriding parameters

    Returns
    -------
    list of lib.Potential, masses

    Note: If an atom has zero eps and nonzero charge, sig and eps are overridden to small nonzero values.
        (Otherwise, U(x) will spuriously go to -inf if the distance between oppositely charged particles is
        brought to 0, which is possible with flexible TIP3P.)
        (Another possible workaround to consider: Use CHARMM parameterization of TIP3P.)
    """

    masses = []

    for p in range(system.getNumParticles()):
        masses.append(value(system.getParticleMass(p)))

    N = len(masses)

    # key type is "List[bp]" rather than "bp", since we may have more than one instance of a given force.
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

            num_atoms = force.getNumParticles()

            charge_params_ = []
            lj_params_ = []

            _charges_of_overridden_particles = []

            for a_idx in range(num_atoms):

                charge, sig, eps = force.getParticleParameters(a_idx)
                charge = value(charge) * np.sqrt(constants.ONE_4PI_EPS0)

                sig = value(sig)
                eps = value(eps)

                # override (sig, eps) if we have eps==0, to avoid singularities in potential and derivatives
                if (eps == 0) and (charge != 0):
                    _charges_of_overridden_particles.append(charge)
                    sig_override = 1e-4  # small radius
                    eps_override = 1e-4  # repulsive

                    msg = f"""Warning:
                        overriding (q={charge}, sig={sig}, eps={eps}) to
                        ({charge}, {sig_override}, {eps_override}) to avoid singularity"""
                    if verbose:
                        print(msg)

                    sig = sig_override
                    eps = eps_override

                # charge_params.append(charge_idx)
                charge_params_.append(charge)
                lj_params_.append((sig, eps))

            # a case that the LJ override could mishandle is if
            if len(_charges_of_overridden_particles) > 0:
                # most favorable pair
                _q_ij = np.max(_charges_of_overridden_particles) * np.min(_charges_of_overridden_particles)
                assert _q_ij < 0, "input contained particles with LJ eps == 0 and charges of opposite signs"

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
                new_sig = value(new_sig)
                new_eps = value(new_eps)

                src_sig = all_sig[src]
                dst_sig = all_sig[dst]

                src_eps = all_eps[src]
                dst_eps = all_eps[dst]
                expected_sig = (src_sig + dst_sig) / 2
                expected_eps = np.sqrt(src_eps * dst_eps)

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

                scale_factors_.append(lj_scale_factor)

                # tbd fix charge_scale_factors using new_cp
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

            # confirm that eps > 0 for any particle with charge != 0
            assert (lj_params[charge_params != 0, 1] > 0).all()

            # optimizations
            nb_params[:, 1] = nb_params[:, 1] / 2
            nb_params[:, 2] = np.sqrt(nb_params[:, 2])

            beta = 2.0  # erfc correction

            # use the same scale factors for electrostatics and lj
            scale_factors = np.stack([scale_factors_, scale_factors_], axis=1)

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
