import os
import numpy as np

from simtk import openmm as mm
from simtk.openmm import app
from simtk.openmm.app import PDBFile
from simtk.openmm.app import forcefield as ff
from simtk import unit

from timemachine import constants

def value(quantity):
    return quantity.value_in_unit_system(unit.md_unit_system)

def deserialize_system(system):
    """
    Deserialize an OpenMM XML file

    Parameters
    ----------
    system: openmm.System
        A system object to be deserialized

    Returns
    -------
    list of energy functions, masses

    """

    masses = []

    for p in range(system.getNumParticles()):
        masses.append(value(system.getParticleMass(p)))

    # this should not be a dict since we may have more than one instance of a given
    # force.
    nrg_fns = []

    nb_charge_params = None
    gb_charge_params = None

    for force in system.getForces():

        if isinstance(force, mm.HarmonicBondForce):
            bond_idxs = []
            # param_idxs = []
            bond_params = []

            for b_idx in range(force.getNumBonds()):
                src_idx, dst_idx, length, k = force.getBondParameters(b_idx)
                length = value(length)
                k = value(k)

                bond_idxs.append([src_idx, dst_idx])
                bond_params.append((k, length))

            bond_idxs = np.array(bond_idxs, dtype=np.int32)
            bond_params = np.array(bond_params, dtype=np.float64)
            nrg_fns.append(("HarmonicBond", (bond_idxs, bond_params)))

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

            nrg_fns.append(("HarmonicAngle", (angle_idxs, angle_params)))

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
            nrg_fns.append(("PeriodicTorsion", (torsion_idxs, torsion_params)))

        if isinstance(force, mm.NonbondedForce):

            num_atoms = force.getNumParticles()

            nb_charge_params = []
            lj_params = []

            for a_idx in range(num_atoms):

                charge, sig, eps = force.getParticleParameters(a_idx)
                charge = value(charge)*np.sqrt(constants.ONE_4PI_EPS0)

                sig = value(sig)
                eps = value(eps)

                # (ytz): this is only necessary if the protein atoms are allowed to be in a separate plane
                # override default amber types
                # if sig == 0 or eps == 0:
                    # sig = 0.1
                    # eps = 0.1

                # charge_idx = insert_parameters(charge, 14)
                # sig_idx = insert_parameters(sig, 10)
                # eps_idx = insert_parameters(eps, 11)

                # nb_charge_params.append(charge_idx)
                nb_charge_params.append(charge)
                lj_params.append((sig, eps))

            nb_charge_params = np.array(nb_charge_params, dtype=np.float64)

            # print("Protein net charge:", np.sum(np.array(global_params)[charge_param_idxs]))
            lj_params = np.array(lj_params, dtype=np.float64)

            # 1 here means we fully remove the interaction
            # 1-2, 1-3
            # scale_full = insert_parameters(1.0, 20)

            # 1-4, remove half of the interaction
            # scale_half = insert_parameters(0.5, 21)

            nb_exclusion_idxs = []
            lj_exclusion_params = []

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

                nb_exclusion_idxs.append([src, dst])

                # sanity check this (expected_eps can be zero), redo this thing

                # the lj_scale factor measures how much we *remove*
                if expected_eps == 0:
                    if new_eps == 0:
                        lj_scale_factor = 1
                    else:
                        raise RuntimeError("Divide by zero in epsilon calculation")
                else:
                    lj_scale_factor = 1 - new_eps/expected_eps

                lj_exclusion_params.append(lj_scale_factor)

                # tbd fix charge_scale_factors using new_cp
                if new_eps != 0:
                    np.testing.assert_almost_equal(expected_sig, new_sig)
                #     np.testing.assert_almost_equal(new_eps/expected_eps, 0.5)

                #     exclusion_params.append(scale_idx_half)
                # else:
                #     exclusion_params.append(scale_idx_full)

            nb_exclusion_idxs = np.array(nb_exclusion_idxs, dtype=np.int32)
            # exclusion_param_idxs = np.array(exclusion_param_idxs, dtype=np.int32)

            nrg_fns.append(("LennardJones", 
                lj_params,
            ))

        if isinstance(force, mm.GBSAOBCForce):

            num_atoms = force.getNumParticles()

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

            gb_params = []
            gb_charge_params = []

            for a_idx in range(num_atoms):
                charge, radius, scale = force.getParticleParameters(a_idx)

                # this needs to be scaled by sqrt(eps0)
                charge = value(charge)*np.sqrt(constants.ONE_4PI_EPS0)
                gb_charge_params.append(charge)

                radius = value(radius)
                gb_params.append((radius, scale))

            gb_params = np.array(gb_params, dtype=np.float64)

            nrg_fns.append(("GBSA", (
                gb_params,
                alpha,                         # alpha
                beta,                          # beta
                gamma,                         # gamma
                dielectric_offset,             # dielectric_offset
                surface_tension,               # surface_tension
                solute_dielectric,             # solute_dielectric
                solvent_dielectric,            # solvent_dieletric
                probe_radius                   # probe_radius
            )))

    # ensure GB charges and NB charges are consistent 
    if gb_charge_params is not None and nb_charge_params is not None:
        np.testing.assert_almost_equal(gb_charge_params, nb_charge_params)

    gb_charge_params = np.array(gb_charge_params)

    nrg_fns.append(('Charges', gb_charge_params))

    charge_exclusion_params = np.array(lj_exclusion_params)

    nrg_fns.append(('Exclusions', (nb_exclusion_idxs, lj_exclusion_params, charge_exclusion_params)))

    return nrg_fns, masses