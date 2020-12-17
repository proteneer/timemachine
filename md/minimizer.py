import numpy as np

from fe import topology

from timemachine.lib import potentials
from timemachine.lib import LangevinIntegrator, custom_ops

from ff.handlers import openmm_deserializer
from ff import Forcefield

def get_romol_conf(mol):
    """Coordinates of mol's 0th conformer, in nanometers"""
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf/10 # from angstroms to nm


def minimize_host_4d(romol, host_system, host_coords, ff, box):
    """
    Insert romol into a host system via 4D decoupling under a Langevin thermostat.
    The ligand coordinates are fixed during this, and only host_coordinates are minimized.

    Parameters
    ----------
    romol: ROMol
        Ligand to be inserted. It must be embedded.

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

    # keep the ligand rigid
    ligand_masses = [a.GetMass()*100000 for a in romol.GetAtoms()]
    combined_masses = np.concatenate([host_masses, ligand_masses])
    ligand_coords = get_romol_conf(romol)
    combined_coords = np.concatenate([host_coords, ligand_coords])
    num_host_atoms = host_coords.shape[0]

    final_potentials = []
    for bp in host_bps:
        if isinstance(bp, potentials.Nonbonded):
            host_p = bp
        else:
            final_potentials.append(bp)

    gbt = topology.BaseTopology(romol, ff)
    hgt = topology.HostGuestTopology(host_p, gbt)

    # setup the parameter handlers for the ligand
    tuples = [
        [hgt.parameterize_harmonic_bond, [ff.hb_handle]],
        [hgt.parameterize_harmonic_angle, [ff.ha_handle]],
        [hgt.parameterize_proper_torsion, [ff.pt_handle]],
        [hgt.parameterize_improper_torsion, [ff.it_handle]],
        [hgt.parameterize_nonbonded, [ff.q_handle, ff.lj_handle]],
    ]

    for fn, handles in tuples:
        params, potential = fn(*[h.params for h in handles])
        final_potentials.append(potential.bind(params))

    seed = 2020

    intg = LangevinIntegrator(
        300.0,
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

    return ctxt.get_x_t()[:num_host_atoms]
