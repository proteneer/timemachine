import warnings
from typing import Optional, Tuple

import numpy as np
import scipy.optimize
from numpy.typing import NDArray
from rdkit import Chem
from simtk import openmm

from timemachine.fe import model_utils, topology
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.fire import fire_descent


class MinimizationWarning(UserWarning):
    pass


def bind_potentials(topo, ff: Forcefield):
    # setup the parameter handlers for the ligand
    ff_params = ff.get_params()
    params_potential_pairs = [
        topo.parameterize_harmonic_bond(ff_params.hb_params),
        topo.parameterize_harmonic_angle(ff_params.ha_params),
        topo.parameterize_periodic_torsion(ff_params.pt_params, ff_params.it_params),
        topo.parameterize_nonbonded(ff_params.q_params, ff_params.lj_params),
    ]
    u_impls = [potential.bind(params).bound_impl(precision=np.float32) for params, potential in params_potential_pairs]

    return u_impls


def fire_minimize(x0: NDArray, u_impls, box: NDArray, lamb_sched: NDArray) -> NDArray:
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
    _, xs, _ = ctxt.multiple_steps(np.linspace(1.0, 0, 1000))

    final_coords = fire_minimize(xs[-1], u_impls, box, np.zeros(50))
    for impl in u_impls:
        du_dx, _, _ = impl.execute(final_coords, box, 0.0)
        norm = np.linalg.norm(du_dx, axis=-1)
        assert np.all(norm < 25000)

    return final_coords[:num_host_atoms]


def equilibrate_host(
    mol: Chem.Mol,
    host_system: openmm.System,
    host_coords: NDArray,
    temperature: float,
    pressure: float,
    ff: Forcefield,
    box: NDArray,
    n_steps: int,
    seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray]:
    """
    Equilibrate a host system given a reference molecule using the MonteCarloBarostat.

    Useful for preparing a host that will be used for multiple FEP calculations using the same reference, IE a starmap.

    Performs the following:
    - Minimize host with rigid mol
    - Minimize host and mol
    - Run n_steps with HMR enabled and MonteCarloBarostat every 5 steps

    Parameters
    ----------
    mol: Chem.Mol
        Ligand for the host to equilibrate with.

    host_system: openmm.System
        OpenMM System representing the host.

    host_coords: np.ndarray
        N x 3 coordinates of the host. units of nanometers.

    temperature: float
        Temperature at which to run the simulation. Units of kelvins.

    pressure: float
        Pressure at which to run the simulation. Units of bars.

    ff: ff.Forcefield
        Wrapper class around a list of handlers.

    box: np.ndarray [3,3]
        Box matrix for periodic boundary conditions. units of nanometers.

    n_steps: int
        Number of steps to run the simulation for.

    seed: int or None
        Value to seed simulation with

    Returns
    -------
    tuple (coords, box)
        Returns equilibrated system coords as well as the box.

    """
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
    ff_params = ff.get_params()
    hb_params, hb_potential = hgt.parameterize_harmonic_bond(ff_params.hb_params)

    params_potential_pairs = [
        (hb_params, hb_potential),
        hgt.parameterize_harmonic_angle(ff_params.ha_params),
        hgt.parameterize_periodic_torsion(ff_params.pt_params, ff_params.it_params),
        hgt.parameterize_nonbonded(ff_params.q_params, ff_params.lj_params),
    ]

    u_impls = [pot.bind(params).bound_impl(precision=np.float32) for params, pot in params_potential_pairs]
    bond_list = get_bond_list(hb_potential)
    combined_masses = model_utils.apply_hmr(combined_masses, bond_list)

    dt = 2.5e-3
    friction = 1.0

    if seed is None:
        seed = np.random.randint(np.iinfo(np.int32).max)

    integrator = LangevinIntegrator(temperature, dt, friction, combined_masses, seed).impl()

    x0 = combined_coords
    v0 = np.zeros_like(x0)

    group_indices = get_group_indices(bond_list)
    barostat_interval = 5
    barostat = MonteCarloBarostat(x0.shape[0], pressure, temperature, group_indices, barostat_interval, seed).impl(
        u_impls
    )

    # Re-minimize with the mol being flexible
    x0 = fire_minimize(x0, u_impls, box, np.ones(50))
    # context components: positions, velocities, box, integrator, energy fxns
    ctxt = custom_ops.Context(x0, v0, box, integrator, u_impls, barostat)

    ctxt.multiple_steps(np.linspace(0.0, 0.0, n_steps))

    return ctxt.get_x_t(), ctxt.get_box()


def get_val_and_grad_fn(bps, box, lamb):
    """
    Convert impls, box, lamb into a function that only takes in coords.

    Parameters
    ----------
    bps: List of BoundPotentials

    box: np.array (3,3)

    lamb: float

    Returns
    -------
    Energy function with gradient
        f: R^(Nx3) -> (R^1, R^Nx3)
    """

    def val_and_grad_fn(coords):
        g = np.zeros_like(coords)
        u = 0.0
        for impl in bps:
            g_bp, _, u_bp = impl.execute(coords, box, lamb)
            g += g_bp
            u += u_bp
        return u, g

    return val_and_grad_fn


def local_minimize(x0, val_and_grad_fn, local_idxs, verbose=True, assert_energy_decreased=True):
    """
    Minimize a local region given selected idxs.

    Parameters:
    -----------
    x0: np.array (N,3)
        Coordinates

    val_and_grad_fn: f: R^(Nx3) -> (R^1, R^Nx3)
        Energy function

    local_idxs: list of int
        Unique idxs we allow to move.

    verbose: bool
        Print internal scipy.optimize warnings + potential energy + gradient norm

    assert_energy_decreased: bool
        Throw an assertion if the energy does not decrease

    Returns
    -------
    Optimized set of coordinates (N,3)

    """

    assert len(local_idxs) == len(set(local_idxs))
    n_local = len(local_idxs)
    n_frozen = len(x0) - n_local

    x_local_shape = (n_local, 3)
    u_0, _ = val_and_grad_fn(x0)

    # deal with overflow, empirically obtained by testing on some real systems.
    guard_threshold = 5e4

    def val_and_grad_fn_local(x_local):
        x_prime = x0.copy()
        x_prime[local_idxs] = x_local
        u_full, grad_full = val_and_grad_fn(x_prime)
        # avoid being trapped when overflows spuriously appear as large negative numbers
        # remove after resolution of https://github.com/proteneer/timemachine/issues/481
        if u_0 - u_full > guard_threshold:
            u_full = np.inf
            grad_full = np.nan * grad_full
        return u_full, grad_full[local_idxs]

    # deals with reshaping from (L,3) -> (Lx3,)
    def val_and_grad_fn_bfgs(x_local_flattened):
        x_local = x_local_flattened.reshape(x_local_shape)
        u, grad_full = val_and_grad_fn_local(x_local)
        return u, grad_full.reshape(-1)

    x_local_0 = x0[local_idxs]
    x_local_0_flat = x_local_0.reshape(-1)

    method = "BFGS"

    if verbose:
        print("-" * 70)
        print(f"performing {method} minimization on {n_local} atoms\n(holding the other {n_frozen} atoms frozen)")
        U_0, grad_0 = val_and_grad_fn_bfgs(x_local_0_flat)
        print(f"U(x_0) = {U_0:.3f}")

    res = scipy.optimize.minimize(
        val_and_grad_fn_bfgs,
        x_local_0_flat,
        method=method,
        jac=True,
        options={"disp": verbose},
    )

    x_local_final_flat = res.x
    x_local_final = x_local_final_flat.reshape(x_local_shape)

    if verbose:
        U_final, grad_final = val_and_grad_fn_bfgs(x_local_final_flat)
        print(f"U(x_final) = {U_final:.3f}")

        # diagnose worst atom
        forces = -grad_final.reshape(x_local_shape)
        per_atom_force_norms = np.linalg.norm(forces, axis=1)
        argmax_local = np.argmax(per_atom_force_norms)
        worst_atom_idx = local_idxs[argmax_local]
        print(f"atom with highest force norm after minimization: {worst_atom_idx}")
        print(f"force(x_final)[{worst_atom_idx}] = {forces[argmax_local]}")

        print("-" * 70)

    x_final = x0.copy()
    x_final[local_idxs] = x_local_final

    u_final, _ = val_and_grad_fn(x_final)

    if assert_energy_decreased:
        assert u_final < u_0, f"u_0: {u_0:.3f}, u_f: {u_final:.3f}"
    elif u_final >= u_0:
        warnings.warn(f"WARNING: Energy did not decrease: u_0: {u_0:.3f}, u_f: {u_final:.3f}", MinimizationWarning)

    return x_final
