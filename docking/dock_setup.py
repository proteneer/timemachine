import numpy as np

from simtk.openmm import app

from ff.handlers import bonded, nonbonded, openmm_deserializer

from timemachine.lib import potentials


def combine_parameters(guest_q, guest_lj, host_qlj):
    guest_qlj = np.concatenate(
        [np.reshape(guest_q, (-1, 1)), np.reshape(guest_lj, (-1, 2))], axis=1
    )
    return np.concatenate([host_qlj, guest_qlj])


def combine_potentials(guest_ff_handlers, guest_mol, host_system, precision):
    """
    This function is responsible for figuring out how to take two separate hamiltonians
    and combining them into one sensible alchemical system.

    Parameters
    ----------

    ff_handlers: list of forcefield handlers
        Small molecule forcefield handlers

    guest_mol: Chem.ROMol
        RDKit molecule

    host_system: openmm.System
        Host system to be deserialized

    precision: np.float32 or np.float64
        Numerical precision of the functional form

    Returns
    -------
    tuple
        Returns a list of lib.potentials objects, combined masses, and a list of
        their corresponding vjp_fns back into the forcefield

    """

    host_potentials, host_masses = openmm_deserializer.deserialize_system(
        host_system, precision, cutoff=1.0
    )
    host_nb_bp = None

    combined_potentials = []

    for bp in host_potentials:
        if isinstance(bp, potentials.Nonbonded):
            # (ytz): hack to ensure we only have one nonbonded term
            assert host_nb_bp is None
            host_nb_bp = bp
        else:
            combined_potentials.append(bp)

    guest_masses = np.array(
        [a.GetMass() for a in guest_mol.GetAtoms()], dtype=np.float64
    )
    num_host_atoms = len(host_masses)
    num_guest_atoms = len(guest_masses)

    combined_masses = np.concatenate([host_masses, guest_masses])

    # guest handlers
    for handle in guest_ff_handlers:
        results = handle.parameterize(guest_mol)
        if isinstance(handle, bonded.HarmonicBondHandler):
            bond_idxs, (bond_params, vjp_fn) = results
            bond_idxs += num_host_atoms
            combined_potentials.append(
                potentials.HarmonicBond(bond_idxs, precision=precision).bind(
                    bond_params
                )
            )

        elif isinstance(handle, bonded.HarmonicAngleHandler):
            angle_idxs, (angle_params, vjp_fn) = results
            angle_idxs += num_host_atoms
            combined_potentials.append(
                potentials.HarmonicAngle(angle_idxs, precision=precision).bind(
                    angle_params
                )
            )

        elif isinstance(handle, bonded.ProperTorsionHandler):
            torsion_idxs, (torsion_params, vjp_fn) = results
            torsion_idxs += num_host_atoms
            combined_potentials.append(
                potentials.PeriodicTorsion(torsion_idxs, precision=precision).bind(
                    torsion_params
                )
            )

        elif isinstance(handle, bonded.ImproperTorsionHandler):
            torsion_idxs, (torsion_params, vjp_fn) = results
            torsion_idxs += num_host_atoms
            combined_potentials.append(
                potentials.PeriodicTorsion(torsion_idxs, precision=precision).bind(
                    torsion_params
                )
            )

        elif isinstance(handle, nonbonded.AM1CCCHandler):
            charge_handle = handle
            guest_charge_params, guest_charge_vjp_fn = results

        elif isinstance(handle, nonbonded.LennardJonesHandler):
            guest_lj_params, guest_lj_vjp_fn = results
            lj_handle = handle

        else:
            print("Warning: skipping handler", handle)
            pass

    # process nonbonded terms
    combined_nb_params = combine_parameters(
        guest_charge_params, guest_lj_params, host_nb_bp.params
    )

    # these vjp_fns take in adjoints of combined_params and returns derivatives
    # appropriate to the underlying handler

    # tbd change scale 14 for electrostatics
    guest_exclusion_idxs, guest_scale_factors = nonbonded.generate_exclusion_idxs(
        guest_mol, scale12=1.0, scale13=1.0, scale14=0.5
    )

    # allow the ligand to be alchemically decoupled
    # a value of one indicates that we allow the atom to be adjusted by the lambda value
    guest_lambda_offset_idxs = np.ones(len(guest_masses), dtype=np.int32)

    # use same scale factors until we modify 1-4s for electrostatics
    guest_scale_factors = np.stack([guest_scale_factors, guest_scale_factors], axis=1)

    combined_lambda_offset_idxs = np.concatenate(
        [host_nb_bp.get_lambda_offset_idxs(), guest_lambda_offset_idxs]
    )
    combined_exclusion_idxs = np.concatenate(
        [host_nb_bp.get_exclusion_idxs(), guest_exclusion_idxs + num_host_atoms]
    )

    # print("EXCLUSIONS", host_nb_bp.get_exclusion_idxs())
    # assert 0
    combined_scales = np.concatenate(
        [host_nb_bp.get_scale_factors(), guest_scale_factors]
    )

    combined_beta = 2.0

    combined_cutoff = 1.0  # nonbonded cutoff

    combined_nb_params = np.asarray(combined_nb_params).copy()

    # print(combined_nb_params[1758:])
    # combined_nb_params[:, 0] = 0.0

    # print(combined_nb_params)

    combined_potentials.append(
        potentials.Nonbonded(
            combined_exclusion_idxs,
            combined_scales,
            combined_lambda_offset_idxs,
            combined_beta,
            combined_cutoff,
            precision=precision,
        ).bind(combined_nb_params)
    )

    return combined_potentials, combined_masses
