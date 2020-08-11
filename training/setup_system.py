import jax
import jax.numpy as jnp
import numpy as np

from simtk.openmm import app

from ff.handlers import bonded, nonbonded, openmm_deserializer
from fe.utils import to_md_units

from timemachine.potentials import jax_utils
from timemachine.potentials import bonded as bonded_utils

def find_protein_pocket_atoms(conf, nha, search_radius):
    """
    Find atoms in the protein that are close to the binding pocket. This simply grabs the
    protein atoms that are within search_radius nm of each ligand atom.

    Parameters
    ----------
    conf: np.array [N,3]
        conformation of the ligand

    nha: int
        number of host atoms

    search_radius: float
        how far we search into the binding pocket.

    """
    ri = np.expand_dims(conf, axis=0)
    rj = np.expand_dims(conf, axis=1)
    dij = jax_utils.distance(ri, rj)
    pocket_atoms = set()

    for l_idx, dists in enumerate(dij[nha:]):
        nns = np.argsort(dists[:nha])
        for p_idx in nns:
            if dists[p_idx] < search_radius:
                pocket_atoms.add(p_idx)


    return list(pocket_atoms)

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

def create_system(
    guest_mol,
    host_pdb,
    handlers,
    search_radius,
    stage):
    """
    Initialize a self-encompassing System object that we can serialize and simulate.

    Parameters
    ----------

    guest_mol: rdkit.ROMol
        guest molecule
        
    host_pdb: openmm.PDBFile
        host system from OpenMM

    handlers: list of timemachine.ops.Gradients
        forcefield handlers used to parameterize the system

    search_radius: float
        how far away we search from the ligand to define the binding pocket atoms.

    stage: int (0 or 1)
        a free energy specific variable that determines how we decouple.
 
    """

    guest_masses = np.array([a.GetMass() for a in guest_mol.GetAtoms()], dtype=np.float64)

    amber_ff = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
    host_system = amber_ff.createSystem(
        host_pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False
    )

    host_fns, host_masses = openmm_deserializer.deserialize_system(host_system)

    num_host_atoms = len(host_masses)
    num_guest_atoms = guest_mol.GetNumAtoms()

    # Name, Args, vjp_fn
    final_gradients = []
    final_vjp_fns = []

    for item in host_fns: 

        if item[0] == 'LennardJones':
            host_lj_params = item[1]
        elif item[0] == 'Charges':
            host_charge_params = item[1]
        elif item[0] == 'GBSA':
            host_gb_params = item[1][0]
            host_gb_props = item[1][1:]
        elif item[0] == 'Exclusions':
            host_exclusions = item[1]
        else:
            final_gradients.append((item[0], item[1]))
            final_vjp_fns.append(None)

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

    for handle in handlers:
        results = handle.parameterize(guest_mol)

        if isinstance(handle, bonded.HarmonicBondHandler):
            bond_idxs, (bond_params, bond_vjp_fn) = results
            bond_idxs += num_host_atoms
            final_gradients.append(("HarmonicBond", (bond_idxs, bond_params)))
            final_vjp_fns.append((bond_vjp_fn))
            # handler_vjps.append(bond_vjp_fn)
        elif isinstance(handle, bonded.HarmonicAngleHandler):
            angle_idxs, (angle_params, angle_vjp_fn) = results
            angle_idxs += num_host_atoms
            final_gradients.append(("HarmonicAngle", (angle_idxs, angle_params)))
            final_vjp_fns.append(angle_vjp_fn)
            # handler_vjps.append(angle_vjp_fn)
        elif isinstance(handle, bonded.ProperTorsionHandler):
            torsion_idxs, (torsion_params, torsion_vjp_fn) = results
            torsion_idxs += num_host_atoms
            final_gradients.append(("PeriodicTorsion", (torsion_idxs, torsion_params)))
            final_vjp_fns.append(torsion_vjp_fn)
            # handler_vjps.append(torsion_vjp_fn)
            # guest_vjp_fns.append(torsion_vjp_fn)
        elif isinstance(handle, bonded.ImproperTorsionHandler):
            torsion_idxs, (torsion_params, torsion_vjp_fn) = results
            torsion_idxs += num_host_atoms
            final_gradients.append(("PeriodicTorsion", (torsion_idxs, torsion_params)))
            final_vjp_fns.append(torsion_vjp_fn)
            # handler_vjps.append(torsion_vjp_fn)
        elif isinstance(handle, nonbonded.LennardJonesHandler):
            guest_lj_params, guest_lj_vjp_fn = results
            combined_lj_params, combined_lj_vjp_fn = concat_with_vjps(host_lj_params, guest_lj_params, None, guest_lj_vjp_fn)
            # final_gradients.append(("LennardJones", (torsion_idxs, torsion_params)))
            # final_vjp_fns.append(combined_lj_vjp_fn)
            # handler_vjps.append(lj_adjoint_fn)
        elif isinstance(handle, nonbonded.SimpleChargeHandler):
            guest_charge_params, guest_charge_vjp_fn = results
            combined_charge_params, combined_charge_vjp_fn = concat_with_vjps(host_charge_params, guest_charge_params, None, guest_charge_vjp_fn)
            # handler_vjps.append(charge_adjoint_fn)
        elif isinstance(handle, nonbonded.GBSAHandler):
            guest_gb_params, guest_gb_vjp_fn = results
            combined_gb_params, combined_gb_vjp_fn = concat_with_vjps(host_gb_params, guest_gb_params, None, guest_gb_vjp_fn)
            # handler_vjps.append(gb_adjoint_fn)
        elif isinstance(handle, nonbonded.AM1BCCHandler):
            # ill defined behavior if both SimpleChargeHandler and AM1Handler is present
            guest_charge_params, guest_charge_vjp_fn = results
            combined_charge_params, combined_charge_vjp_fn = concat_with_vjps(host_charge_params, guest_charge_params, None, guest_charge_vjp_fn)
            # handler_vjps.append(gb_adjoint_fn)
        elif isinstance(handle, nonbonded.AM1CCCHandler):
            guest_charge_params, guest_charge_vjp_fn = results
            combined_charge_params, combined_charge_vjp_fn = concat_with_vjps(host_charge_params, guest_charge_params, None, guest_charge_vjp_fn)
            # handler_vjps.append(gb_adjoint_fn)
        else:
            raise Exception("Unknown Handler", handle)

    # (use the below vjps for correctness)
    # combined_charge_params, charge_adjoint_fn = concat_with_vjps(host_charge_params, guest_charge_params, None, guest_charge_vjp_fn)
    # combined_lj_params, lj_adjoint_fn = concat_with_vjps(host_lj_params, guest_lj_params, None, guest_lj_vjp_fn)
    # combined_gb_params, gb_adjoint_fn = concat_with_vjps(host_gb_params, guest_gb_params, None, guest_gb_vjp_fn)

    host_conf = []
    for x,y,z in host_pdb.positions:
        host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])
    host_conf = np.array(host_conf)

    conformer = guest_mol.GetConformer(0)
    mol_a_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    mol_a_conf = mol_a_conf/10 # convert to md_units

    x0 = np.concatenate([host_conf, mol_a_conf]) # combined geometry
    v0 = np.zeros_like(x0)

    pocket_atoms = find_protein_pocket_atoms(x0, num_host_atoms, search_radius)

    N_C = num_host_atoms + num_guest_atoms
    N_A = num_host_atoms

    # if stage == 0:
    #     combined_lambda_plane_idxs = np.zeros(N_C, dtype=np.int32)
    #     combined_lambda_offset_idxs = np.zeros(N_C, dtype=np.int32)
    #     combined_lambda_offset_idxs[num_host_atoms:] = 1
    #     combined_lambda_offset_idxs[pocket_atoms] = 1

    #     # grouping for vdw terms
    #     combined_lambda_group_idxs = np.ones(N_C, dtype=np.int32)
    #     combined_lambda_group_idxs[num_host_atoms:] = 2
    #     combined_lambda_group_idxs[pocket_atoms] = 3
    # elif stage == 1:
    #     combined_lambda_plane_idxs = np.zeros(N_C, dtype=np.int32)
    #     combined_lambda_plane_idxs[num_host_atoms:] = 1
    #     combined_lambda_plane_idxs[pocket_atoms] = 1

    #     combined_lambda_offset_idxs = np.zeros(N_C, dtype=np.int32)
    #     combined_lambda_offset_idxs[num_host_atoms:] = 1 # push out ligand from binding pocket

    #     # grouping for vdw terms
    #     combined_lambda_group_idxs = np.ones(N_C, dtype=np.int32)
    #     combined_lambda_group_idxs[num_host_atoms:] = 2
    # else:
    #     assert 0

    cutoff = 100000.0

    # final_gradients.append((
    #     'LennardJones', (
    #     np.asarray(combined_lj_params),
    #     combined_exclusion_idxs,
    #     combined_lj_exclusion_scales,
    #     combined_lambda_plane_idxs,
    #     combined_lambda_offset_idxs,
    #     combined_lambda_group_idxs,
    #     cutoff
    #     )
    # ))
    # final_vjp_fns.append((combined_lj_vjp_fn))

    # set up lambdas for electrostatics

    if stage == 0:
        # assert 0
        # combined_lambda_plane_idxs = np.zeros(N_C, dtype=np.int32)
        # combined_lambda_offset_idxs = np.zeros(N_C, dtype=np.int32)
        # combined_lambda_offset_idxs[num_host_atoms:] = 1

        # turn on restraints
        combined_lambda_plane_idxs = np.zeros(N_C, dtype=np.int32)
        combined_lambda_offset_idxs = np.zeros(N_C, dtype=np.int32)
    elif stage == 1:
        # combined_lambda_plane_idxs = np.zeros(N_C, dtype=np.int32)
        # combined_lambda_plane_idxs[num_host_atoms:] = 1
        # combined_lambda_offset_idxs = np.zeros(N_C, dtype=np.int32)

        # decouple
        combined_lambda_plane_idxs = np.zeros(N_C, dtype=np.int32)
        combined_lambda_offset_idxs = np.zeros(N_C, dtype=np.int32)
        combined_lambda_offset_idxs[num_host_atoms:] = 1
    else:
        assert 0

    # final_gradients.append((
    #     'Electrostatics', (
    #     np.asarray(combined_charge_params),
    #     combined_exclusion_idxs,
    #     combined_charge_exclusion_scales,
    #     combined_lambda_plane_idxs,
    #     combined_lambda_offset_idxs,
    #     cutoff
    #     )
    # ))
    # final_vjp_fns.append((combined_charge_vjp_fn))

    final_gradients.append((
        'Nonbonded', (
        np.asarray(combined_charge_params),
        np.asarray(combined_lj_params),
        combined_exclusion_idxs,
        combined_charge_exclusion_scales,
        combined_lj_exclusion_scales,
        combined_lambda_plane_idxs,
        combined_lambda_offset_idxs,
        cutoff
        )
    ))
    final_vjp_fns.append((combined_charge_vjp_fn, combined_lj_vjp_fn))

    final_gradients.append((
        'GBSA', (
        np.asarray(combined_charge_params),
        np.asarray(combined_gb_params),
        combined_lambda_plane_idxs,
        combined_lambda_offset_idxs,
        *host_gb_props,
        cutoff,
        cutoff
        )
    ))
    final_vjp_fns.append((combined_charge_vjp_fn, combined_gb_vjp_fn))

    ligand_idxs = np.arange(N_A, N_C, dtype=np.int32)


    r1, r2, r3 = [1019-1, 1020-1, 1035-1]
    l1, l2, l3 = [1766-1, 1769-1, 1795-1]


    # K_r/2 * [|r3 - l1| - r_aA0]^2 +
    # + K_thetaA/2 * [angle(r2,r3,l1) - theta_A0]^2 +
    # + K_thetaB/2 * [angle(r3,l1,l2) - theta_B0]^2 +
    # + K_phiA/2 * hav(dihedral(r1,r2,r3,l1) - phi_A0) * 2 +
    # + K_phiB/2 * hav(dihedral(r2,r3,l1,l2) - phi_B0) * 2 +
    # + K_phiC/2 * hav(dihedral(r3,l1,l2,l3) - phi_C0) * 2

    def bond(a, b):
        ci = np.expand_dims(x0[a], axis=0)
        cj = np.expand_dims(x0[b], axis=0)
        return jax_utils.distance(ci, cj, None)[0]

    def angle(a, b, c):
        ci = np.expand_dims(x0[a], axis=0)
        cj = np.expand_dims(x0[b], axis=0)
        ck = np.expand_dims(x0[c], axis=0)

        vij = ci - cj
        vjk = ck - cj

        top = np.sum(np.multiply(vij, vjk), -1)
        bot = np.linalg.norm(vij, axis=-1)*np.linalg.norm(vjk, axis=-1)

        tb = top/bot
        angle = np.arccos(tb)
        return angle[0]

    def torsion(a,b,c,d):
        ci = np.expand_dims(x0[a], axis=0)
        cj = np.expand_dims(x0[b], axis=0)
        ck = np.expand_dims(x0[c], axis=0)
        cl = np.expand_dims(x0[d], axis=0)
        return bonded_utils.signed_torsion_angle(ci, cj, ck, cl)[0]

    boresch_bond_idxs = [[r3,l1]]
    boresch_angle_idxs = [[r2,r3,l1], [r3,l1,l2]]
    boresch_torsion_idxs = [[r1,r2,r3,l1], [r2,r3,l1,l2], [r3,l1,l2,l3]]

    boresch_bond_params = np.asarray([[1000.0, bond(*x)*1.1] for x in boresch_bond_idxs])
    boresch_angle_params = np.asarray([[100.0, angle(*x)] for x in boresch_angle_idxs])
    boresch_torsion_params = np.array([[100.0, torsion(*x)] for x in boresch_torsion_idxs])



    # # print(angle(r2,r3,l1), angle(r3,l1,l2))
    # # print(torsion(r1,r2,r3,l1), torsion(r2,r3,l1,l2), torsion(r3,l1,l2,l3))

    # assert 0


    # avg_xi = np.mean(x0[ligand_idxs], axis=0)
    # avg_xj = np.mean(x0[pocket_atoms], axis=0)
    # ctr_dij = np.sqrt(np.sum((avg_xi - avg_xj)**2))

    # print("centroid distance", ctr_dij)

    combined_masses = np.concatenate([host_masses, guest_masses])

    # print("ligand idxs", ligand_idxs)
    # print("pocket idxs", pocket_atoms)

    # restraints
    if stage == 0:
        lamb_flag = 1
        lamb_offset = 0
    if stage == 1:
        lamb_flag = 0
        lamb_offset = 1

    final_gradients.append((
        'BoreschLikeRestraint', (
            boresch_bond_idxs,
            boresch_angle_idxs,
            boresch_torsion_idxs,
            boresch_bond_params,
            boresch_angle_params,
            boresch_torsion_params,
            lamb_flag,
            lamb_offset,
        )
    ))

    # final_gradients.append((
    #     'CentroidRestraint', (
    #         ligand_idxs,
    #         pocket_atoms,
    #         combined_masses,
    #         1000.0,
    #         ctr_dij,
    #         lamb_flag,
    #         lamb_offset
    #     )
    # ))

    final_vjp_fns.append(lambda x: None)

    return x0, combined_masses, final_gradients, final_vjp_fns
