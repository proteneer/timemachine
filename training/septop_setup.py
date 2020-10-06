import jax
import jax.numpy as jnp
import numpy as np

from ff.handlers import bonded, nonbonded, openmm_deserializer
from timemachine.lib import potentials


def combine_coordinates(
    host_coords,
    guest_mol_a,
    guest_mol_b):

    host_conf = np.array(host_coords)

    conformer_a = guest_mol_a.GetConformer(0)
    guest_conf_a = np.array(conformer_a.GetPositions(), dtype=np.float64)
    guest_conf_a = guest_conf_a/10 # convert to md_units

    conformer_b = guest_mol_b.GetConformer(0)
    guest_conf_b = np.array(conformer_b.GetPositions(), dtype=np.float64)
    guest_conf_b = guest_conf_b/10 # convert to md_units

    return np.concatenate([host_conf, guest_conf_a, guest_conf_b]) # combined geometry

def nonbonded_vjps(
    guest_a_q, guest_a_q_vjp_fn,
    guest_a_lj, guest_a_lj_vjp_fn,
    guest_b_q, guest_b_q_vjp_fn,
    guest_b_lj, guest_b_lj_vjp_fn,
    host_qlj):
    """

    Returns
    -------
    (P+A+B, 3)
        Parameterized system with host atoms at the front.

    (guest_q_vjp_fn, guest_lj_vjp_fn)
        Chain rule'd vjps to enable combined adjoints to backprop into handler params.

    """ 


    def combine_parameters(
        guest_a_q,
        guest_a_lj,
        guest_b_q,
        guest_b_lj,
        host_qlj):

        guest_q = jnp.concatenate([guest_a_q, guest_b_q])
        guest_lj = jnp.concatenate([guest_a_lj, guest_b_lj])

        guest_qlj = jnp.concatenate([
            jnp.reshape(guest_q, (-1, 1)),
            jnp.reshape(guest_lj, (-1, 2))
        ], axis=1)

        return jnp.concatenate([host_qlj, guest_qlj])

    combine_parameters(
        guest_a_q, guest_a_lj,
        guest_b_q, guest_b_lj,
        host_qlj
    )

    combined_qlj, combined_vjp_fn = jax.vjp(
        combine_parameters,
        guest_a_q, guest_a_lj,
        guest_b_q, guest_b_lj,
        host_qlj
    )

    def g_a_q_vjp_fn(x):
        combined_adjoint = combined_vjp_fn(x)[0]
        return guest_a_q_vjp_fn(combined_adjoint)

    def g_b_q_vjp_fn(x):
        combined_adjoint = combined_vjp_fn(x)[1]
        return guest_b_q_vjp_fn(combined_adjoint)

    def g_a_lj_vjp_fn(x):
        combined_adjoint = combined_vjp_fn(x)[2]
        return guest_a_lj_vjp_fn(combined_adjoint)

    def g_b_lj_vjp_fn(x):
        combined_adjoint = combined_vjp_fn(x)[3]
        return guest_b_lj_vjp_fn(combined_adjoint)

    return combined_qlj, (g_a_q_vjp_fn, g_b_q_vjp_fn, g_a_lj_vjp_fn, g_b_lj_vjp_fn)


# def concat_with_vjps(p_a, p_b, vjp_a, vjp_b):
#     """
#     Returns the combined parameters p_c, and a vjp_fn that can take in adjoint with shape
#     of p_c and returns adjoints of primitives of p_a and p_b.

#     i.e. 
#        vjp_a            
#     A' -----> A 
#                 \ vjp_c
#                  +-----> C
#        vjp_b    /
#     B' -----> B

#     """

#     # concat_fn = functools.partial()

#     p_c, vjp_c = jax.vjp(jnp.concatenate, [p_a, p_b])
#     adjoints = np.random.randn(*p_c.shape)

#     def adjoint_fn(p_c):
#         ad_a, ad_b = vjp_c(p_c)[0]
#         if vjp_a is not None:
#             ad_a = vjp_a(ad_a)
#         else:
#             ad_a = None

#         if vjp_b is not None:
#             ad_b = vjp_b(ad_b)
#         else:
#             ad_b = None

#         return ad_b

#     return np.asarray(p_c), adjoint_fn

def combine_potentials(
    ff_handlers,
    guest_mol_a,
    guest_mol_b,
    host_system,
    precision):
    """
    This function is responsible for figuring out how to take two separate hamiltonians
    and combining them into one sensible alchemical system.

    Parameters
    ----------

    ff_handlers: list of forcefield handlers
        Small molecule forcefield handlers

    guest_mol_a: Chem.ROMol
        RDKit molecule

    guest_mol_b: Chem.ROMol
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
        host_system,
        precision,
        cutoff=1.0
    )

    host_nb_bp = None

    combined_potentials = []
    combined_vjp_fns = []

    for bp in host_potentials:
        if isinstance(bp, potentials.Nonbonded):
            # (ytz): hack to ensure we only have one nonbonded term
            assert host_nb_bp is None
            host_nb_bp = bp
        else:
            combined_potentials.append(bp)
            combined_vjp_fns.append([])

    guest_a_masses = np.array([a.GetMass() for a in guest_mol_a.GetAtoms()], dtype=np.float64)
    guest_b_masses = np.array([b.GetMass() for b in guest_mol_b.GetAtoms()], dtype=np.float64)

    num_host_atoms = len(host_masses)

    combined_masses = np.concatenate([host_masses, guest_a_masses, guest_b_masses])

    # guest_charge_handles = []
    guest_charge_params = []
    guest_charge_vjp_fns = []
    guest_lj_params = []
    guest_lj_vjp_fns = []

    for handle in ff_handlers:

        offset_atoms = num_host_atoms

        for guest_mol in [guest_mol_a, guest_mol_b]:

            results = handle.parameterize(guest_mol)
            if isinstance(handle, bonded.HarmonicBondHandler):
                bond_idxs, (bond_params, vjp_fn) = results
                bond_idxs += offset_atoms
                combined_potentials.append(potentials.HarmonicBond(bond_idxs, precision=precision).bind(bond_params))
                combined_vjp_fns.append([(handle, vjp_fn)])
            elif isinstance(handle, bonded.HarmonicAngleHandler):
                angle_idxs, (angle_params, vjp_fn) = results
                angle_idxs += offset_atoms
                combined_potentials.append(potentials.HarmonicAngle(angle_idxs, precision=precision).bind(angle_params))
                combined_vjp_fns.append([(handle, vjp_fn)])
            elif isinstance(handle, bonded.ProperTorsionHandler):
                torsion_idxs, (torsion_params, vjp_fn) = results
                torsion_idxs += offset_atoms
                combined_potentials.append(potentials.PeriodicTorsion(torsion_idxs, precision=precision).bind(torsion_params))
                combined_vjp_fns.append([(handle, vjp_fn)])
            elif isinstance(handle, bonded.ImproperTorsionHandler):
                torsion_idxs, (torsion_params, vjp_fn) = results
                torsion_idxs += offset_atoms
                combined_potentials.append(potentials.PeriodicTorsion(torsion_idxs, precision=precision).bind(torsion_params))
                combined_vjp_fns.append([(handle, vjp_fn)])
            elif isinstance(handle, nonbonded.AM1CCCHandler):
                charge_handle = handle
                # guest_charge_params, guest_charge_vjp_fn = results

                guest_charge_params.append(results[0])
                guest_charge_vjp_fns.append(results[1])


            elif isinstance(handle, nonbonded.LennardJonesHandler):
                # guest_lj_params, guest_lj_vjp_fn = results
                lj_handle = handle

                guest_lj_params.append(results[0])
                guest_lj_vjp_fns.append(results[1])

            else:
                print("Warning: skipping handler", handle)
                pass

            offset_atoms += guest_mol.GetNumAtoms()

    # def nonbonded_vjps(
    # guest_a_q, guest_a_q_vjp_fn,
    # guest_a_lj, guest_a_lj_vjp_fn,
    # guest_b_q, guest_b_q_vjp_fn,
    # guest_b_lj, guest_b_lj_vjp_fn,
    # host_qlj):
    # """

    # process nonbonded terms
    combined_nb_params, (charge_a_vjp_fn, charge_b_vjp_fn, lj_a_vjp_fn, lj_b_vjp_fn) = nonbonded_vjps(
        guest_charge_params[0], guest_charge_vjp_fns[0],
        guest_lj_params[0], guest_lj_vjp_fns[0],
        guest_charge_params[1], guest_charge_vjp_fns[1],
        guest_lj_params[1], guest_lj_vjp_fns[1],
        host_nb_bp.params
    )

    # (ytz): we can probably combined the vjps
    combined_vjp_fns.append([
        (charge_handle, charge_a_vjp_fn),
        (charge_handle, charge_b_vjp_fn),
        (lj_handle, lj_a_vjp_fn),
        (lj_handle, lj_b_vjp_fn)
    ])

    # tbd change scale 14 for electrostatics
    guest_a_exclusion_idxs, guest_a_scale_factors = nonbonded.generate_exclusion_idxs(
        guest_mol_a,
        scale12=1.0,
        scale13=1.0,
        scale14=0.5
    )

    guest_b_exclusion_idxs, guest_b_scale_factors = nonbonded.generate_exclusion_idxs(
        guest_mol_b,
        scale12=1.0,
        scale13=1.0,
        scale14=0.5
    )

    # allow the ligand to be alchemically decoupled
    # a value of one indicates that we allow the atom to be adjusted by the lambda value
    # we need to minimize the system some how


    # use same scale factors until we modify 1-4s for electrostatics
    guest_scale_factors = np.concatenate([guest_a_scale_factors, guest_b_scale_factors])
    guest_scale_factors = np.stack([guest_scale_factors, guest_scale_factors], axis=1)

    guest_lambda_offset_idxs = np.ones(len(guest_b_masses), dtype=np.int32) 
    combined_lambda_offset_idxs = np.zeros(num_host_atoms + len(guest_a_masses) + len(guest_b_masses), dtype=np.int32)
    combined_lambda_offset_idxs[(num_host_atoms + len(guest_a_masses)):] = 1
    # combined_lambda_offset_idxs[num_host_atoms:] = 1

    offset = num_host_atoms
    combined_exclusion_idxs = np.concatenate([host_nb_bp.get_exclusion_idxs(), guest_a_exclusion_idxs + offset])
    offset = num_host_atoms + guest_mol_a.GetNumAtoms()
    combined_exclusion_idxs = np.concatenate([combined_exclusion_idxs, guest_b_exclusion_idxs + offset])

    mol_a_idxs = num_host_atoms + np.arange(guest_mol_a.GetNumAtoms())
    mol_b_idxs = num_host_atoms + guest_mol_a.GetNumAtoms() + np.arange(guest_mol_b.GetNumAtoms())

    mol_mol_exclusions = []
    mol_mol_scales = []
    for i in mol_a_idxs:
        for j in mol_b_idxs:
            mol_mol_exclusions.append((i,j))
            mol_mol_scales.append((1.0, 1.0))

    combined_exclusion_idxs = np.concatenate([combined_exclusion_idxs, mol_mol_exclusions])
    combined_scales = np.concatenate([host_nb_bp.get_scale_factors(), guest_scale_factors, mol_mol_scales])

    combined_beta = 2.0
    combined_cutoff = 1.0 # nonbonded cutoff

    combined_potentials.append(potentials.Nonbonded(
        combined_exclusion_idxs.astype(np.int32),
        combined_scales,
        combined_lambda_offset_idxs,
        combined_beta,
        combined_cutoff,
        precision=precision).bind(combined_nb_params))

    # add CoM restraints
    kb = 100.0
    b0 = 0.0 # keep the CoMs as close to zero as possible

    NT = len(combined_masses)

    for i in mol_a_idxs:
        assert i < NT

    for j in mol_b_idxs:
        assert j < NT

    empty_params = np.array([], dtype=np.float64)
    combined_potentials.append(potentials.CentroidRestraint(
        np.array(mol_a_idxs, dtype=np.int32),
        np.array(mol_b_idxs, dtype=np.int32),
        np.array(combined_masses, dtype=np.float64),
        kb,
        b0,
        precision=precision).bind(empty_params)
    )
    combined_vjp_fns.append([])


    # print(len(combined_potentials), len(combined_vjp_fns))

    # for idx, p in enumerate(combined_potentials):
        # print(idx, p)

    return combined_potentials, combined_masses, combined_vjp_fns
