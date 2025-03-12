import numpy as np
import openmm as mm
from openmm import unit

from timemachine import constants, potentials
from timemachine.ff.handlers.utils import canonicalize_bond


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
        src, dst, new_q, new_sig, new_eps = force.getExceptionParameters(a_idx)
        desired_q = value(new_q) * constants.ONE_4PI_EPS0
        desired_sig = value(new_sig)
        desired_eps = value(new_eps)

        src_sig = all_sig[src]
        dst_sig = all_sig[dst]

        src_eps = all_eps[src]
        dst_eps = all_eps[dst]
        initial_sig = (src_sig + dst_sig) / 2
        initial_eps = np.sqrt(src_eps * dst_eps)

        src_q = charge_params[src]
        dst_q = charge_params[dst]

        initial_q = src_q * dst_q

        exclusion_idxs_.append([src, dst])
        # the lj_scale factor measures how much we *remove*
        if initial_eps == 0:
            if desired_eps == 0:
                lj_scale_factor = 1
            else:
                raise RuntimeError("No LJ scaling factor possible to arrive at desired_eps")
        else:
            lj_scale_factor = 1 - desired_eps / initial_eps

        if initial_q == 0:
            if desired_q == 0:
                q_scale_factor = 1
            else:
                raise RuntimeError("No ES scaling factor possible to arrive at desired_q")
        else:
            q_scale_factor = 1 - desired_q / initial_q  # noqa

        scale_factors_.append(
            (
                lj_scale_factor,  # TODO: investigate FEP performance regression when set to OFF value
                lj_scale_factor,
            )
        )

        # check combining rules for sigmas are consistent
        if desired_eps != 0:
            np.testing.assert_almost_equal(initial_sig, desired_sig)

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


def deserialize_system(system: mm.System, cutoff: float) -> tuple[list[potentials.BoundPotential], list[float]]:
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

    """

    bond = angle = proper = improper = nonbonded = None

    masses = []

    for p in range(system.getNumParticles()):
        masses.append(value(system.getParticleMass(p)))

    N = len(masses)

    # this should not be a dict since we may have more than one instance of a given
    # force.

    # process bonds and angles first to instantiate bond_idxs and angle_idxs
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
            bond = potentials.HarmonicBond(bond_idxs).bind(bond_params)

        if isinstance(force, mm.HarmonicAngleForce):
            angle_idxs_ = []
            angle_params_ = []

            for a_idx in range(force.getNumAngles()):
                src_idx, mid_idx, dst_idx, angle, k = force.getAngleParameters(a_idx)
                angle = value(angle)
                k = value(k)

                angle_idxs_.append([src_idx, mid_idx, dst_idx])
                angle_params_.append((k, angle, 0.0))  # 0.0 is for epsilon

            angle_idxs = np.array(angle_idxs_, dtype=np.int32)
            angle_params = np.array(angle_params_, dtype=np.float64)
            angle = potentials.HarmonicAngle(angle_idxs).bind(angle_params)

    for force in system.getForces():
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

            # (ytz): split torsion into proper and impropers, if both angles are present
            # then it's a proper torsion, otherwise it's an improper torsion
            canonical_angle_idxs = set(canonicalize_bond(tuple(idxs)) for idxs in angle_idxs)

            proper_idxs = []
            proper_params = []
            improper_idxs = []
            improper_params = []

            for idxs, params in zip(torsion_idxs, torsion_params):
                i, j, k, l = idxs
                angle_ijk = canonicalize_bond((i, j, k))
                angle_jkl = canonicalize_bond((j, k, l))
                if angle_ijk in canonical_angle_idxs and angle_jkl in canonical_angle_idxs:
                    proper_idxs.append(idxs)
                    proper_params.append(params)
                elif angle_ijk not in canonical_angle_idxs and angle_jkl not in canonical_angle_idxs:
                    assert 0
                else:
                    # xor case imply improper
                    improper_idxs.append(idxs)
                    improper_params.append(params)

            proper = potentials.PeriodicTorsion(np.array(proper_idxs)).bind(np.array(proper_params))
            improper = potentials.PeriodicTorsion(np.array(improper_idxs)).bind(np.array(improper_params))

        if isinstance(force, mm.NonbondedForce):
            nb_params, exclusion_idxs, beta, scale_factors = deserialize_nonbonded_force(force, N)

            nonbonded = potentials.Nonbonded(N, exclusion_idxs, scale_factors, beta, cutoff).bind(nb_params)

    assert bond
    assert angle
    assert nonbonded

    if proper is None:
        proper = potentials.PeriodicTorsion(np.array([], dtype=np.int32).reshape(-1, 4)).bind(
            np.array([], dtype=np.float64).reshape(-1, 3)
        )
    if improper is None:
        improper = potentials.PeriodicTorsion(np.array([], dtype=np.int32).reshape(-1, 4)).bind(
            np.array([], dtype=np.float64).reshape(-1, 3)
        )

    bps = [bond, angle, proper, improper, nonbonded]

    return bps, masses
