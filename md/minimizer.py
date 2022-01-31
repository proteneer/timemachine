import numpy as np

from fe import topology

from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops

from timemachine.ff.handlers import openmm_deserializer
from timemachine.ff import Forcefield
from fe import model_utils
from fe.utils import get_romol_conf
from md.barostat.utils import get_group_indices, get_bond_list

from rdkit import Chem
from simtk import openmm

from md.fire import fire_descent


def bind_potentials(topo, ff):
    # setup the parameter handlers for the ligand
    tuples = [
        [topo.parameterize_harmonic_bond, [ff.hb_handle]],
        [topo.parameterize_harmonic_angle, [ff.ha_handle]],
        [topo.parameterize_periodic_torsion, [ff.pt_handle, ff.it_handle]],
        [topo.parameterize_nonbonded, [ff.q_handle, ff.lj_handle]],
    ]

    u_impls = []

    for fn, handles in tuples:
        params, potential = fn(*[h.params for h in handles])
        bp = potential.bind(params)
        u_impls.append(bp.bound_impl(precision=np.float32))
    return u_impls


def fire_minimize(x0: np.ndarray, u_impls, box: np.ndarray, lamb_sched: np.array) -> np.ndarray:
    """
    Minimize coordinates using the FIRE algorithm

    Parameters
    ----------
    coords: np.ndarray
        N x 3 coordinates. units of nanometers.

    u_impls: list of bound impls of potentials

    box: np.ndarray [3,3]
        Box matrix for periodic boundary conditions. units of nanometers.

    lamb_sched: np.array [N]
        Array of lambda for each step of the optimization.

    Returns
    -------
    np.ndarray
        Minimized coords.

    """

    def force(coords, lamb: float = 1.0, **kwargs):
        forces = np.zeros_like(coords)
        for impl in u_impls:
            du_dx, _, _ = impl.execute(coords, box, lamb)
            forces -= du_dx
        return forces

    def shift(d, dr, **kwargs):
        return d + dr

    init, f = fire_descent(force, shift)
    opt_state = init(x0, lamb=lamb_sched[0])
    for lamb in lamb_sched[1:]:
        opt_state = f(opt_state, lamb=lamb)
    return np.asarray(opt_state.position)


def minimize_host_4d(mols, host_system, host_coords, ff, box, mol_coords=None) -> np.ndarray:
    """
    Insert mols into a host system via 4D decoupling using Fire minimizer at lambda=1.0,
    0 Kelvin Langevin integration at a sequence of lambda from 1.0 to 0.0, and Fire minimizer again at lambda=0.0

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

    mol_coords: list of np.ndarray
        Pre-specify a list of mol coords. Else use the mol.GetConformer(0)

    Returns
    -------
    np.ndarray
        This returns minimized host_coords.

    """

    assert box.shape == (3, 3)

    host_bps, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)

    num_host_atoms = host_coords.shape[0]

    if len(mols) == 1:
        top = topology.BaseTopology(mols[0], ff)
    elif len(mols) == 2:
        top = topology.DualTopologyMinimization(mols[0], mols[1], ff)
    else:
        raise ValueError("mols must be length 1 or 2")

    mass_list = [np.array(host_masses)]
    conf_list = [np.array(host_coords)]
    for mol in mols:
        # mass increase is to keep the ligand fixed
        mass_list.append(np.array([a.GetMass() * 100000 for a in mol.GetAtoms()]))

    if mol_coords is not None:
        for mc in mol_coords:
            conf_list.append(mc)
    else:
        for mol in mols:
            conf_list.append(get_romol_conf(mol))

    combined_masses = np.concatenate(mass_list)
    combined_coords = np.concatenate(conf_list)

    hgt = topology.HostGuestTopology(host_bps, top)

    u_impls = bind_potentials(hgt, ff)

    # this value doesn't matter since we will turn off the noise.
    seed = 0

    intg = LangevinIntegrator(0.0, 1.5e-3, 1.0, combined_masses, seed).impl()

    x0 = combined_coords
    v0 = np.zeros_like(x0)

    x0 = fire_minimize(x0, u_impls, box, np.ones(50))
    # context components: positions, velocities, box, integrator, energy fxns
    ctxt = custom_ops.Context(x0, v0, box, intg, u_impls)
    ctxt.multiple_steps(np.linspace(1.0, 0, 1000))

    final_coords = fire_minimize(ctxt.get_x_t(), u_impls, box, np.zeros(50))
    for impl in u_impls:
        du_dx, _, _ = impl.execute(final_coords, box, 0.0)
        norm = np.linalg.norm(du_dx, axis=-1)
        assert np.all(norm < 25000)

    return final_coords[:num_host_atoms]


def equilibrate_host(
    mol: Chem.Mol,
    host_system: openmm.System,
    host_coords: np.array,
    temperature: float,
    pressure: float,
    ff: Forcefield,
    box: np.array,
    n_steps,
):

    # insert mol into the binding pocket.
    host_bps, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)

    min_host_coords = minimize_host_4d([mol], host_system, host_coords, ff, box)

    ligand_masses = [a.GetMass() for a in mol.GetAtoms()]
    ligand_coords = get_romol_conf(mol)

    combined_masses = np.concatenate([host_masses, ligand_masses])
    combined_coords = np.concatenate([min_host_coords, ligand_coords])

    top = topology.BaseTopology(mol, ff)
    hgt = topology.HostGuestTopology(host_bps, top)

    # setup the parameter handlers for the ligand
    tuples = [
        [hgt.parameterize_harmonic_bond, [ff.hb_handle]],
        [hgt.parameterize_harmonic_angle, [ff.ha_handle]],
        [hgt.parameterize_periodic_torsion, [ff.pt_handle, ff.it_handle]],
        [hgt.parameterize_nonbonded, [ff.q_handle, ff.lj_handle]],
    ]

    u_impls = []
    bound_potentials = []

    for fn, handles in tuples:
        params, potential = fn(*[h.params for h in handles])
        bp = potential.bind(params)
        bound_potentials.append(bp)
        u_impls.append(bp.bound_impl(precision=np.float32))

    bond_list = get_bond_list(bound_potentials[0])
    combined_masses = model_utils.apply_hmr(combined_masses, bond_list)

    dt = 2.5e-3
    friction = 1.0

    seed = 0

    integrator = LangevinIntegrator(temperature, dt, friction, combined_masses, seed).impl()

    x0 = combined_coords
    v0 = np.zeros_like(x0)

    group_indices = get_group_indices(bond_list)
    barostat_interval = 5
    barostat = MonteCarloBarostat(x0.shape[0], pressure, temperature, group_indices, barostat_interval, seed).impl(
        u_impls
    )

    x0 = fire_minimize(x0, u_impls, box, np.ones(50))
    # context components: positions, velocities, box, integrator, energy fxns
    ctxt = custom_ops.Context(x0, v0, box, integrator, u_impls, barostat)

    ctxt.multiple_steps(np.linspace(0.0, 0.0, n_steps))

    return ctxt.get_x_t(), ctxt.get_box()
