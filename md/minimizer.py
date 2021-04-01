import numpy as np

from fe import topology

from jankmachine.lib import potentials
from jankmachine.lib import LangevinIntegrator, custom_ops

from ff.handlers import openmm_deserializer
from ff import Forcefield

def get_romol_conf(mol):
    """Coordinates of mol's 0th conformer, in nanometers"""
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf/10 # from angstroms to nm


def minimize_host_4d(mols, host_system, host_coords, ff, box):
    """
    Insert mols into a host system via 4D decoupling using a 0 Kelvin Langevin integrator.

    The ligand coordinates are fixed during this, and only host_coords are minimized.

    Parameters
    ----------
    mols: list of Chem.Mol
        Ligands to be inserted. This must be of length 1 or 2 for now.

    host_system: openmm.System
        OpenMM System representing the host

    host_coords: np.ndarray
        N x 3 coordinates of the host. units of nanometers.

    ff: ff.Forcefield
        Wrapper class around a list of handlers

    box: np.ndarray [3,3]
        Box matrix for periodic boundary conditions. units of nanometers.

    Returns
    -------
    np.ndarray
        This returns minimized host_coords.

    """

    host_bps, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)

    num_host_atoms = host_coords.shape[0]

    if len(mols) == 1:
        top = topology.BaseTopology(mols[0], ff)
    elif len(mols) == 2:
        top = topology.DualTopology(mols[0], mols[1], ff)
    else:
        raise ValueError("mols must be length 1 or 2")

    mass_list = [np.array(host_masses)]
    conf_list = [np.array(host_coords)]
    for mol in mols:
        # mass increase is to keep the ligand fixed
        mass_list.append(np.array([a.GetMass()*100000 for a in mol.GetAtoms()]))
        conf_list.append(get_romol_conf(mol))

    combined_masses = np.concatenate(mass_list)
    combined_coords = np.concatenate(conf_list)

    hgt = topology.HostGuestTopology(host_bps, top)

    # setup the parameter handlers for the ligand
    tuples = [
        [hgt.parameterize_harmonic_bond, [ff.hb_handle]],
        [hgt.parameterize_harmonic_angle, [ff.ha_handle]],
        [hgt.parameterize_periodic_torsion, [ff.pt_handle, ff.it_handle]],
        [hgt.parameterize_nonbonded, [ff.q_handle, ff.lj_handle]],
    ]

    final_potentials = []

    for fn, handles in tuples:
        params, potential = fn(*[h.params for h in handles])
        final_potentials.append(potential.bind(params))

    # this value doesn't matter since we will turn off the noise.
    seed = 0

    intg = LangevinIntegrator(
        0.0,
        1.5e-3,
        1.0,
        combined_masses,
        seed
    ).impl()

    x0 = combined_coords
    v0 = np.zeros_like(x0)

    u_impls = []

    for bp in final_potentials:
        fn = bp.bound_impl(precision=np.float32)
        u_impls.append(fn)

    # context components: positions, velocities, box, integrator, energy fxns
    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        intg,
        u_impls
    )

    for lamb in np.linspace(1.0, 0, 1000):
        ctxt.step(lamb)

    final_coords = ctxt.get_x_t()

    for impl in u_impls:
        du_dx, _, _ = impl.execute(final_coords, box, 0.0)
        assert np.all(np.linalg.norm(du_dx, axis=-1) < 25000)

    return final_coords[:num_host_atoms]
