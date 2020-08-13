import numpy as np

from simtk.openmm import app

from fe.utils import to_md_units
from ff.handlers import bonded, nonbonded, openmm_deserializer

def create_system(
    guest_mol,
    host_pdb,
    handlers):
    """
    Initialize a self-encompassing System object that we can serialize and simulate.

    Parameters
    ----------

    guest_mol: rdkit.ROMol
        guest molecule
        
    host_pdb: openmm.PDBFile
        host system from OpenMM

    handlers: list of timemachine.ops.Gradients
        forcefield handlers used to parameterize the small molecule
 
    Returns
    -------
    3-tuple
        x0, combined_masses, final_gradients

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
            bond_idxs, (bond_params, _) = results
            bond_idxs += num_host_atoms
            final_gradients.append(("HarmonicBond", (bond_idxs, bond_params)))
        elif isinstance(handle, bonded.HarmonicAngleHandler):
            angle_idxs, (angle_params, _) = results
            angle_idxs += num_host_atoms
            final_gradients.append(("HarmonicAngle", (angle_idxs, angle_params)))
        elif isinstance(handle, bonded.ProperTorsionHandler):
            torsion_idxs, (torsion_params, _) = results
            torsion_idxs += num_host_atoms
            final_gradients.append(("PeriodicTorsion", (torsion_idxs, torsion_params)))
        elif isinstance(handle, bonded.ImproperTorsionHandler):
            torsion_idxs, (torsion_params, _) = results
            torsion_idxs += num_host_atoms
            final_gradients.append(("PeriodicTorsion", (torsion_idxs, torsion_params)))
        elif isinstance(handle, nonbonded.LennardJonesHandler):
            guest_lj_params, _ = results
            combined_lj_params = np.concatenate([host_lj_params, guest_lj_params])
        elif isinstance(handle, nonbonded.SimpleChargeHandler):
            guest_charge_params, _ = results
            combined_charge_params = np.concatenate([host_charge_params, guest_charge_params])
        elif isinstance(handle, nonbonded.GBSAHandler):
            guest_gb_params, _ = results
            combined_gb_params = np.concatenate([host_gb_params, guest_gb_params])
        elif isinstance(handle, nonbonded.AM1BCCHandler):
            guest_charge_params, _ = results
            combined_charge_params = np.concatenate([host_charge_params, guest_charge_params])
        elif isinstance(handle, nonbonded.AM1CCCHandler):
            guest_charge_params, _ = results
            combined_charge_params = np.concatenate([host_charge_params, guest_charge_params])
        else:
            raise Exception("Unknown Handler", handle)

    host_conf = []
    for x,y,z in host_pdb.positions:
        host_conf.append([to_md_units(x),to_md_units(y),to_md_units(z)])
    host_conf = np.array(host_conf)

    conformer = guest_mol.GetConformer(0)
    mol_a_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    mol_a_conf = mol_a_conf/10 # convert to md_units

    center = np.mean(mol_a_conf, axis=0)

    mol_a_conf -= center

    from scipy.stats import special_ortho_group
    mol_a_conf = np.matmul(mol_a_conf, special_ortho_group.rvs(3))
    mol_a_conf += center

    # assert 0



    x0 = np.concatenate([host_conf, mol_a_conf]) # combined geometry
    v0 = np.zeros_like(x0)

    N_C = num_host_atoms + num_guest_atoms
    N_A = num_host_atoms

    cutoff = 100000.0

    combined_lambda_plane_idxs = np.zeros(N_C, dtype=np.int32)
    combined_lambda_offset_idxs = np.zeros(N_C, dtype=np.int32)
    combined_lambda_offset_idxs[num_host_atoms:] = 1

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

    combined_masses = np.concatenate([host_masses, guest_masses])

    return x0, combined_masses, final_gradients
