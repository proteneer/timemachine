import jax
import jax.numpy as jnp
import numpy as np

from ff.handlers import bonded, nonbonded, openmm_deserializer
from timemachine.lib import potentials

def combine_coordinates(
    host_coords,
    guest_mol):

    host_conf = np.array(host_coords)
    conformer = guest_mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    guest_conf = guest_conf/10 # convert to md_units

    return np.concatenate([host_conf, guest_conf]) # combined geometry

def concat_with_vjps(p_a, p_b, vjp_a, vjp_b):
    """
    Returns the combined parameters p_c, and a vjp_fn that can take in adjoint with shape
    of p_c and returns adjoints of primitives of p_a and p_b.

    i.e. 
       vjp_a            
    A' -----> A 
                \ vjp_c
                 +-----> C
       vjp_b    /
    B' -----> B

    """
    p_c, vjp_c = jax.vjp(jnp.concatenate, [p_a, p_b])
    adjoints = np.random.randn(*p_c.shape)

    def adjoint_fn(p_c):
        ad_a, ad_b = vjp_c(p_c)[0]
        if vjp_a is not None:
            ad_a = vjp_a(ad_a)
        else:
            ad_a = None

        if vjp_b is not None:
            ad_b = vjp_b(ad_b)
        else:
            ad_b = None

        return ad_b[0]

    return p_c, adjoint_fn

def combine_potentials(
    ff_handlers,
    guest_mol,
    host_system,
    precision):
    """
    Parameters
    ----------

    ff_handlers: list of forcefield handlers
        Small molecule forcefield handlers

    guest_mol: Chem.ROMol
        RDKit molecule

    host_system: openmm.System
        Protein system

    precision: np.float32 or np.float64
        Numerical precision of the functional form

    Returns:
    --------
    tuple
        Returns a list of lib.potentials objects and a list of their corresponding vjp_fns
        back into the forcefield

    """

    host_fns, host_masses = openmm_deserializer.deserialize_system(host_system)

    guest_masses = np.array([a.GetMass() for a in guest_mol.GetAtoms()], dtype=np.float64)

    num_guest_atoms = len(guest_masses)
    num_host_atoms = len(host_masses)

    combined_potentials = []
    combined_vjp_fns = []

    for item in host_fns: 
        if item[0] == 'LennardJones':
            host_lj_params = item[1]
        elif item[0] == 'Charges':
            host_charge_params = item[1]
        elif item[0] == 'Exclusions':
            host_exclusions = item[1]
        else:
            combined_potentials.append((item[0], item[1]))
            combined_vjp_fns.append(None)

    guest_exclusion_idxs, guest_scales = nonbonded.generate_exclusion_idxs(
        guest_mol,
        scale12=1.0,
        scale13=1.0,
        scale14=0.5
    )

    guest_exclusion_idxs += num_host_atoms
    guest_lj_exclusion_scales = guest_scales
    guest_charge_exclusion_scales = guest_scales

    host_exclusion_idxs = host_exclusions[0]
    host_lj_exclusion_scales = host_exclusions[1]
    host_charge_exclusion_scales = host_exclusions[2]

    combined_exclusion_idxs = np.concatenate([host_exclusion_idxs, guest_exclusion_idxs])
    combined_lj_exclusion_scales = np.concatenate([host_lj_exclusion_scales, guest_lj_exclusion_scales])
    combined_charge_exclusion_scales = np.concatenate([host_charge_exclusion_scales, guest_charge_exclusion_scales])

    N_C = num_host_atoms + num_guest_atoms
    N_A = num_host_atoms

    cutoff = 100000.0

    combined_lambda_plane_idxs = np.zeros(N_C, dtype=np.int32)
    combined_lambda_offset_idxs = np.zeros(N_C, dtype=np.int32)
    combined_lambda_offset_idxs[num_host_atoms:] = 1

    combined_masses = np.concatenate([host_masses, guest_masses])

    for handle in ff_handlers:
        results = handle.parameterize(guest_mol)

        if isinstance(handle, bonded.HarmonicBondHandler):
            bond_idxs, (bond_params, vjp_fn) = results
            bond_idxs += num_host_atoms
            # bind potentials
            combined_potentials.append(potentials.HarmonicBond(bond_idxs, precision=precision).bind_impl(bond_params))
            combined_vjp_fns.append((handle, vjp_fn))
        elif isinstance(handle, bonded.HarmonicAngleHandler):
            angle_idxs, (angle_params, vjp_fn) = results
            angle_idxs += num_host_atoms
            combined_potentials.append(potentials.HarmonicAngle(angle_idxs, precision=precision).bind_impl(angle_params))
            combined_vjp_fns.append((handle, vjp_fn))
        elif isinstance(handle, bonded.ProperTorsionHandler):
            torsion_idxs, (torsion_params, vjp_fn) = results
            torsion_idxs += num_host_atoms
            combined_potentials.append(potentials.PeriodicTorsion(torsion_idxs, precision=precision).bind_impl(torsion_params))
            combined_vjp_fns.append((handle, vjp_fn))
        elif isinstance(handle, bonded.ImproperTorsionHandler):
            torsion_idxs, (torsion_params, vjp_fn) = results
            torsion_idxs += num_host_atoms
            combined_potentials.append(potentials.PeriodicTorsion(torsion_idxs, precision=precision).bind_impl(torsion_params))
            combined_vjp_fns.append((handle, vjp_fn))
        elif isinstance(handle, nonbonded.LennardJonesHandler):
            guest_lj_params, guest_lj_vjp_fn = results
            combined_lj_params, vjp_fn = concat_with_vjps(
                host_lj_params,
                guest_lj_params,
                None,
                guest_lj_vjp_fn
            )

            combined_lj_params = np.asarray(combined_lj_params)

            combined_potentials.append(potentials.LennardJones(
                combined_exclusion_idxs,
                combined_lj_exclusion_scales,
                combined_lambda_plane_idxs,
                combined_lambda_offset_idxs,
                cutoff,
                precision=precision).bind_impl(combined_lj_params))

        elif isinstance(handle, nonbonded.AM1CCCHandler):
            guest_charge_params, guest_charge_vjp_fn = results
            combined_charge_params, vjp_fn = concat_with_vjps(
                host_charge_params,
                guest_charge_params,
                None,
                guest_charge_vjp_fn
            )

            combined_charge_params = np.asarray(combined_charge_params)

            beta = 2.0
            combined_potentials.append(potentials.Electrostatics(
                combined_exclusion_idxs,
                combined_charge_exclusion_scales,
                combined_lambda_plane_idxs,
                combined_lambda_offset_idxs,
                beta,
                cutoff,
                precision=precision).bind_impl(combined_charge_params))

            combined_vjp_fns.append((handle, vjp_fn))
        else:
            print("skipping", handle)
            pass

    return combined_potentials, combined_vjp_fns
