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

def nonbonded_vjps(
    guest_q, guest_q_vjp_fn,
    guest_lj, guest_lj_vjp_fn,
    host_qlj):
    """
    Parameters
    ----------


    guest_q: [L, 1]
    guest_q_vjp_fn: f: R^L -> R^L_Q
    guest_lj: [L, 2]
    guest_lj_vjp_fn: f: R^(Lx2) -> R^L_LJ

    """ 

    def combine_parameters(guest_q, guest_lj, host_qlj):
        guest_qlj = jnp.concatenate([
            jnp.reshape(guest_q, (-1, 1)),
            jnp.reshape(guest_lj, (-1, 2))
        ], axis=1)

        return jnp.concatenate([host_qlj, guest_qlj])


    combined_qlj, combined_vjp = jax.vjp(combine_parameters, guest_q, guest_lj, host_qlj)

    # compose the vjp_fns
    guest_q_vjp_fn = lambda x: guest_q_vjp_fn(combined_vjp_fn(x))
    guest_lj_vjp_fn = lambda x: guest_lj_vjp_fn(combined_vjp_fn(x))
    # host_qlj_vjp_fn = lambda x: host_qlj_vjp_fn(combined_vjp_fn(x))

    return combined_qlj, (guest_q_vjp_fn, guest_lj_vjp_fn)


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

    # concat_fn = functools.partial()

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

        return ad_b

    return np.asarray(p_c), adjoint_fn

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

    host_potentials, host_masses = openmm_deserializer.deserialize_system(
        host_system,
        precision,
        cutoff=1.0
    )

    # ensure that electrostatic and lennard jones terms are singletons
    host_nb_bp = None
    # host_es_bp = None

    combined_potentials = []
    combined_vjp_fns = []

    for bp in host_potentials:
        if isinstance(bp, potentials.Nonbonded):
            assert host_nb_bp is None
            host_nb_bp = bp
        # elif isinstance(bp, potentials.Electrostatics):
            # assert host_es_bp is None
            # host_es_bp = bp
        else:
            combined_potentials.append(bp)
            combined_vjp_fns.append([])

    guest_masses = np.array([a.GetMass() for a in guest_mol.GetAtoms()], dtype=np.float64)

    num_guest_atoms = len(guest_masses)
    num_host_atoms = len(host_masses)

    combined_masses = np.concatenate([host_masses, guest_masses])

    # print(ff_handlers)

    # assert 0

    for handle in ff_handlers:

        print("WORKING ON handle", handle)
        results = handle.parameterize(guest_mol)
        if isinstance(handle, bonded.HarmonicBondHandler):
            bond_idxs, (bond_params, vjp_fn) = results
            bond_idxs += num_host_atoms
            combined_potentials.append(potentials.HarmonicBond(bond_idxs, precision=precision).bind(bond_params))
            combined_vjp_fns.append([(handle, vjp_fn)])
        elif isinstance(handle, bonded.HarmonicAngleHandler):
            angle_idxs, (angle_params, vjp_fn) = results
            angle_idxs += num_host_atoms
            combined_potentials.append(potentials.HarmonicAngle(angle_idxs, precision=precision).bind(angle_params))
            combined_vjp_fns.append([(handle, vjp_fn)])
        elif isinstance(handle, bonded.ProperTorsionHandler):
            torsion_idxs, (torsion_params, vjp_fn) = results
            torsion_idxs += num_host_atoms
            combined_potentials.append(potentials.PeriodicTorsion(torsion_idxs, precision=precision).bind(torsion_params))
            combined_vjp_fns.append([(handle, vjp_fn)])
        elif isinstance(handle, bonded.ImproperTorsionHandler):
            torsion_idxs, (torsion_params, vjp_fn) = results
            torsion_idxs += num_host_atoms
            combined_potentials.append(potentials.PeriodicTorsion(torsion_idxs, precision=precision).bind(torsion_params))
            combined_vjp_fns.append([(handle, vjp_fn)])
        elif isinstance(handle, nonbonded.AM1CCCHandler):
            charge_handle = handle
            guest_charge_params, guest_charge_vjp_fn = results
            # combined_charge_params, vjp_fn = concat_with_vjps(
            #     host_es_bp.params,
            #     guest_charge_params,
            #     None,
            #     guest_charge_vjp_fn
            # )

            # combined_exclusion_idxs = np.concatenate([host_es_bp.get_exclusion_idxs(), guest_exclusion_idxs + num_host_atoms])
            # combined_scales = np.concatenate([host_es_bp.get_scale_factors(), guest_scale_factors])
            # combined_lambda_offset_idxs = np.concatenate([host_es_bp.get_lambda_plane_idxs(), guest_lambda_offset_idxs])
            # combined_beta = 2.0

            # combined_potentials.append(potentials.Electrostatics(
            #     combined_exclusion_idxs,
            #     combined_scales,
            #     combined_lambda_offset_idxs,
            #     combined_beta,
            #     combined_cutoff,
            #     precision=precision).bind(combined_charge_params))

            # combined_vjp_fns.append((handle, vjp_fn))
        elif isinstance(handle, nonbonded.LennardJonesHandler):
            guest_lj_params, guest_lj_vjp_fn = results
            lj_handle = handle
            # combined_lj_params, vjp_fn = concat_with_vjps(
            #     host_lj_bp.params,
            #     guest_lj_params,
            #     None,
            #     guest_lj_vjp_fn
            # )

            # TBD FIX ME

            # combined_exclusion_idxs = np.concatenate([host_lj_bp.get_exclusion_idxs(), guest_exclusion_idxs + num_host_atoms])
            # combined_scales = np.concatenate([host_lj_bp.get_scale_factors(), guest_scale_factors])
            # combined_lambda_offset_idxs = np.concatenate([host_lj_bp.get_lambda_plane_idxs(), guest_lambda_offset_idxs])

            # combined_potentials.append(potentials.LennardJones(
            #     combined_exclusion_idxs,
            #     combined_scales,
            #     combined_lambda_offset_idxs,
            #     combined_cutoff,
            #     precision=precision).bind(combined_lj_params))

            # combined_vjp_fns.append((handle, vjp_fn))
        else:
            print("Warning: skipping handle", handle)
            pass

    # process nonbonded terms
    combined_nb_params, (charge_vjp_fn, lj_vjp_fn) = nonbonded_vjps(
        guest_charge_params, guest_charge_vjp_fn,
        guest_lj_params, guest_lj_vjp_fn,
        host_nb_bp.params
    )

    # these vjp_fns take in adjoints of combined_params and returns derivatives
    # appropriate to the underlying handler
    combined_vjp_fns.append([(charge_handle, charge_vjp_fn), (lj_handle, lj_vjp_fn)])

    # tbd change scale 14 for electrostatics
    guest_exclusion_idxs, guest_scale_factors = nonbonded.generate_exclusion_idxs(
        guest_mol,
        scale12=1.0,
        scale13=1.0,
        scale14=0.5
    )

    # allow the ligand to be alchemically decoupled
    guest_lambda_offset_idxs = np.ones(len(guest_masses), dtype=np.int32) 

    # use same scale factors until we modify 1-4s for electrostatics
    guest_scale_factors = np.stack([guest_scale_factors, guest_scale_factors], axis=1)

    combined_lambda_offset_idxs = np.concatenate([host_nb_bp.get_lambda_offset_idxs(), guest_lambda_offset_idxs])
    combined_exclusion_idxs = np.concatenate([host_nb_bp.get_exclusion_idxs(), guest_exclusion_idxs + num_host_atoms])
    combined_scales = np.concatenate([host_nb_bp.get_scale_factors(), guest_scale_factors])
    combined_beta = 2.0

    combined_cutoff = 1.0 # nonbonded cutoff


    combined_potentials.append(potentials.Nonbonded(
        combined_exclusion_idxs,
        combined_scales,
        combined_lambda_offset_idxs,
        combined_beta,
        combined_cutoff,
        precision=precision).bind(combined_nb_params))


    return combined_potentials, combined_masses, combined_vjp_fns
