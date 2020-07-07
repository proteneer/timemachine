import jax
import jax.numpy as jnp
import numpy as np

from simtk.openmm import app

from ff.handlers import bonded, nonbonded, openmm_deserializer
from ff.handlers.deserialize import deserialize
from fe.utils import to_md_units


from timemachine.integrator import langevin_coefficients
from timemachine.potentials import jax_utils

class System():

    def __init__(self, x0, v0, gradients, integrator):
        # fully contained class that allows simulations to be run forward
        # and backward
        self.x0 = x0
        self.v0 = v0
        self.gradients = gradients
        self.integrator = integrator


class Integrator():

    def __init__(self, steps, dt, temperature, friction, masses, lamb, seed):

        equilibrium_steps = 2000

        ca, cbs, ccs = langevin_coefficients(
            temperature,
            dt,
            friction,
            masses
        )

        complete_cas = np.ones(steps)*ca
        complete_dts = np.concatenate([
            np.linspace(0, dt, equilibrium_steps),
            np.ones(steps-equilibrium_steps)*dt
        ])

        self.dts = complete_dts
        self.cas = complete_cas
        self.cbs = -cbs
        self.ccs = ccs
        self.lambs = np.zeros(steps) + lamb
        self.seed = seed

def setup_core_restraints(
    k,
    alpha,
    count,
    conf,
    nha,
    core_atoms,
    stage):
    """
    Setup core restraints

    Parameters
    ----------
    k: float
        Force constant of each restraint

    count: int
        Number of host atoms we restrain each guest_mol to

    nha: int
        Number of host atoms

    core_atoms: list of int
        atoms we're restraining. This is indexed by the total number of atoms in the system.

    stage: 0,1,2
        0 - attach restraint
        1 - decouple
        2 - detach restraint

    """
    ri = np.expand_dims(conf, axis=0)
    rj = np.expand_dims(conf, axis=1)
    dij = jax_utils.distance(ri, rj)
    all_nbs = []

    bond_idxs = []
    bond_params = []

    for l_idx, dists in enumerate(dij[nha:]):
        if l_idx in core_atoms:
            nns = np.argsort(dists[:nha])

            # restrain to 10 nearby atoms
            for p_idx in nns[:count]:
                a = alpha
                b = dists[p_idx]
                bond_params.append((k, b, a))
                bond_idxs.append([l_idx + nha, p_idx])

    bond_idxs = np.array(bond_idxs, dtype=np.int32)
    bond_params = np.array(bond_params, dtype=np.float64)

    B = bond_idxs.shape[0]

    # w = lambda*lambda_flags
    # w = 0 implies that restraints are on
    # w = +inf/-inf implies that restraints are off
    if stage == 0:
        lambda_flags = np.ones(B, dtype=np.int32)
    elif stage == 1:
        # fully interacting
        lambda_flags = np.zeros(B, dtype=np.int32)
    elif stage == 2:
        lambda_flags = np.ones(B, dtype=np.int32)


    return ('Restraint', (
        bond_idxs,
        bond_params,
        lambda_flags
    ))

def create_system(
    guest_mol,
    host_pdb,
    forcefield,
    stage,
    core_atoms,
    restr_force,
    restr_alpha,
    restr_count):
    """
    Initialize a self-encompassing System object that we can serialize and simulate.

    Parameters
    ----------

    guest_mol: rdkit.ROMol

    protein: openmm.System

    """
    # host_system = protein_system

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


    final_gradients = []

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

    # print("Ligand A Name:", a_name)

    ff_raw = open(forcefield, "r").read()
    handlers = deserialize(ff_raw)

    guest_exclusion_idxs, guest_scales = nonbonded.generate_exclusion_idxs(
        guest_mol,
        scale12=1.0,
        scale13=1.0,
        scale14=0.5
    )

    # offset
    guest_exclusion_idxs += num_host_atoms
    # print(guest_exclusion_idxs)
    # print(guest_scales)
    # assert 0


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
        if isinstance(handle, bonded.HarmonicAngleHandler):
            angle_idxs, (angle_params, angle_vjp_fn) = results
            angle_idxs += num_host_atoms
            final_gradients.append(("HarmonicAngle", (angle_idxs, angle_params)))
        elif isinstance(handle, bonded.ProperTorsionHandler):
            torsion_idxs, (torsion_params, torsion_vjp_fn) = results
            torsion_idxs += num_host_atoms
            final_gradients.append(("PeriodicTorsion", (torsion_idxs, torsion_params)))
            # guest_vjp_fns.append(torsion_vjp_fn)
        elif isinstance(handle, bonded.ImproperTorsionHandler):
            torsion_idxs, (torsion_params, torsion_vjp_fn) = results
            torsion_idxs += num_host_atoms
            final_gradients.append(("PeriodicTorsion", (torsion_idxs, torsion_params)))
        elif isinstance(handle, nonbonded.LennardJonesHandler):
            guest_lj_params, guest_lj_vjp_fn = results
        elif isinstance(handle, nonbonded.SimpleChargeHandler):
            guest_charge_params, guest_charge_vjp_fn = results
        elif isinstance(handle, nonbonded.GBSAHandler):
            guest_gb_params, guest_gb_vjp_fn = results

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

            return ad_a, ad_b

        return p_c, adjoint_fn

    combined_charge_params, charge_adjoint_fn = concat_with_vjps(host_charge_params, guest_charge_params, None, guest_charge_vjp_fn)
    combined_lj_params, lj_adjoint_fn = concat_with_vjps(host_lj_params, guest_lj_params, None, guest_lj_vjp_fn)
    combined_gb_params, gb_adjoint_fn = concat_with_vjps(host_gb_params, guest_gb_params, None, guest_gb_vjp_fn)

    # WIP
    N_C = num_host_atoms + num_guest_atoms
    N_A = num_host_atoms

    if stage == 0:
        combined_lambda_plane_idxs = np.zeros(N_C, dtype=np.int32)
        combined_lambda_offset_idxs = np.zeros(N_C, dtype=np.int32)
    elif stage == 1:
        combined_lambda_plane_idxs = np.zeros(N_C, dtype=np.int32)
        combined_lambda_offset_idxs = np.zeros(N_C, dtype=np.int32)
        combined_lambda_offset_idxs[N_A:] = 1
    elif stage == 2:
        combined_lambda_plane_idxs = np.zeros(N_C, dtype=np.int32)
        combined_lambda_plane_idxs[N_A:] = 1
        combined_lambda_offset_idxs = np.zeros(N_C, dtype=np.int32)

    # assert 0

    cutoff = 100000.0

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

    host_conf = []
    for x,y,z in host_pdb.positions:
        host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])
    host_conf = np.array(host_conf)

    conformer = guest_mol.GetConformer(0)
    mol_a_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    mol_a_conf = mol_a_conf/10 # convert to md_units

    x0 = np.concatenate([host_conf, mol_a_conf]) # combined geometry
    v0 = np.zeros_like(x0)

    # build restraints using the coordinates
    final_gradients.append(setup_core_restraints(
        restr_force,
        restr_alpha,
        restr_count,
        x0,
        num_host_atoms,
        core_atoms,
        stage=stage
    ))

    # for f in final_gradients:
        # print("FOOBAR", f[0])

    # assert 0

    # temperature = 300
    # dt = 1.5e-3
    # friction = 40

    combined_masses = np.concatenate([host_masses, guest_masses])

    # integrator = Integrator(dt, temperature, friction, combined_masses)
    # ca, cbs, ccs = langevin_coefficients(
    #     temperature,
    #     dt,
    #     friction,
    #     combined_masses
    # )

    # cbs *= -1

    # print("Integrator coefficients:")
    # print("ca", ca)
    # print("cbs", cbs)
    # print("ccs", ccs)

    return x0, combined_masses, final_gradients
    # return System(x0, v0, final_gradients, integrator)