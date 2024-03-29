import warnings
from typing import Iterable, List, Optional, Sequence, Tuple

import jax
import numpy as np
import scipy.optimize
from numpy.typing import NDArray
from rdkit import Chem

from timemachine.constants import BOLTZ, DEFAULT_TEMP, MAX_FORCE_NORM
from timemachine.fe import model_utils, topology
from timemachine.fe.free_energy import HostConfig
from timemachine.fe.utils import get_mol_masses, get_romol_conf, set_romol_conf
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops
from timemachine.md.barker import BarkerProposal
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.fire import fire_descent
from timemachine.potentials import BoundPotential, HarmonicBond, Potential, SummedPotential


class MinimizationWarning(UserWarning):
    pass


class MinimizationError(Exception):
    pass


def check_force_norm(forces: NDArray, threshold: float = MAX_FORCE_NORM):
    """raise MinimizationError if the force on any atom exceeds MAX_FORCE_NORM"""
    per_atom_force_norms = np.linalg.norm(forces, axis=-1)

    if (per_atom_force_norms > threshold).any():
        bad_inds = np.where(per_atom_force_norms > threshold)[0]
        max_atom_force_norm = np.max(per_atom_force_norms)
        message = f"""
        Minimization failed to reduce large forces below threshold:
            max |frc| = {max_atom_force_norm} > {threshold}
            {len(bad_inds)} / {len(forces)} atoms exceed threshold
        """
        raise MinimizationError(message)


def parameterize_system(topo, ff: Forcefield, lamb: float) -> Tuple[List[Potential], List[NDArray]]:
    # setup the parameter handlers for the ligand
    ff_params = ff.get_params()
    params_potential_pairs = [
        topo.parameterize_harmonic_bond(ff_params.hb_params),
        topo.parameterize_harmonic_angle(ff_params.ha_params),
        topo.parameterize_periodic_torsion(ff_params.pt_params, ff_params.it_params),
        topo.parameterize_nonbonded(
            ff_params.q_params,
            ff_params.q_params_intra,
            ff_params.q_params_solv,
            ff_params.lj_params,
            ff_params.lj_params_intra,
            ff_params.lj_params_solv,
            lamb,
        ),
    ]
    return [pot for (_, pot) in params_potential_pairs], [params for (params, _) in params_potential_pairs]


def flatten_params(params: List[NDArray]) -> NDArray:
    return np.concatenate([p.reshape(-1) for p in params])


def summed_potential_bound_impl_from_potentials_and_params(
    potentials: List[Potential], params: List[NDArray]
) -> custom_ops.BoundPotential:
    flat_params = flatten_params(params)
    return SummedPotential(potentials, params).bind(flat_params).to_gpu(precision=np.float32).bound_impl


def fire_minimize(x0: NDArray, u_impls: Sequence[custom_ops.BoundPotential], box: NDArray, n_steps: int) -> NDArray:
    """
    Minimize coordinates using the FIRE algorithm

    Parameters
    ----------
    coords: np.ndarray
        N x 3 coordinates. units of nanometers.

    u_impls: list of bound impls of potentials

    box: np.ndarray [3,3]
        Box matrix for periodic boundary conditions. units of nanometers.

    n_steps: int
        Number of steps

    Returns
    -------
    np.ndarray
        Minimized coords.

    """

    def force(coords):
        forces = np.zeros_like(coords)
        for impl in u_impls:
            du_dx, _ = impl.execute(coords, box, compute_u=False)
            forces -= du_dx
        return forces

    def shift(d, dr):
        return d + dr

    init, f = fire_descent(force, shift)
    opt_state = init(x0)
    for _ in range(n_steps):
        opt_state = f(opt_state)
    return np.asarray(opt_state.position)


def minimize_host_4d(
    mols: List[Chem.Mol],
    host_config: HostConfig,
    ff: Forcefield,
    mol_coords: Optional[List[NDArray]] = None,
    windows: int = 50,
    n_steps_per_window: int = 50,
) -> np.ndarray:
    """
    Insert mols into a host system via 4D decoupling using Fire minimizer at lambda=1.0,
    0 Kelvin Langevin integration at a sequence of lambda from 1.0 to 0.0, and Fire minimizer again at lambda=0.0

    The ligand coordinates are fixed during this, and only host_coords are minimized.

    Parameters
    ----------
    mols: list of Chem.Mol
        Ligands to be inserted. This must be of length 1 or 2 for now.

    host_config: HostConfig
        Represents the host system.

    ff: ff.Forcefield
        Wrapper class around a list of handlers

    mol_coords: list of np.ndarray
        Pre-specify a list of mol coords. Else use the mol.GetConformer(0)

    windows: integer
        Number of lambda windows to lower the mols into the host via 4D decoupling

    n_steps_per_window: integer
        Number of steps to evaluate at each window

    Returns
    -------
    np.ndarray
        This returns minimized host_coords.

    """
    box = host_config.box
    assert box.shape == (3, 3)

    host_bps, host_masses = openmm_deserializer.deserialize_system(host_config.omm_system, cutoff=1.2)

    num_host_atoms = host_config.conf.shape[0]

    if len(mols) == 1:
        top = topology.BaseTopology(mols[0], ff)
    elif len(mols) == 2:
        top = topology.DualTopologyMinimization(mols[0], mols[1], ff)
    else:
        raise ValueError("mols must be length 1 or 2")

    mass_list = [np.array(host_masses)]
    conf_list = [np.array(host_config.conf)]
    for mol in mols:
        # mass increase is to keep the ligand fixed
        mass_list.append(get_mol_masses(mol) * 100000)

    if mol_coords is not None:
        for mc in mol_coords:
            conf_list.append(mc)
    else:
        for mol in mols:
            conf_list.append(get_romol_conf(mol))

    combined_masses = np.concatenate(mass_list)
    combined_coords = np.concatenate(conf_list)

    hgt = topology.HostGuestTopology(host_bps, top, host_config.num_water_atoms)

    # this value doesn't matter since we will turn off the noise.
    seed = 0

    intg = LangevinIntegrator(0.0, 1.5e-3, 1.0, combined_masses, seed).impl()

    x0 = combined_coords
    v0 = np.zeros_like(x0)

    potentials, params = parameterize_system(hgt, ff, 1.0)
    u_impl = summed_potential_bound_impl_from_potentials_and_params(potentials, params)
    bound_impls = [u_impl]
    x = fire_minimize(x0, bound_impls, box, n_steps_per_window)

    # No need to reconstruct the context, just change the bound potential params. Allows
    # for preserving the velocities between windows
    ctxt = custom_ops.Context(x, v0, box, intg, bound_impls)
    for lamb in np.linspace(1.0, 0, windows):
        _, params = parameterize_system(hgt, ff, lamb)
        u_impl.set_params(flatten_params(params))
        xs, _ = ctxt.multiple_steps(n_steps_per_window)
        x = xs[-1]

    final_coords = fire_minimize(x, bound_impls, box, n_steps_per_window)
    for impl in bound_impls:
        du_dx, _ = impl.execute(final_coords, box, compute_u=False)
        check_force_norm(-du_dx)

    return final_coords[:num_host_atoms]


def make_host_du_dx_fxn(
    mols: List[Chem.Mol], host_config: HostConfig, ff: Forcefield, mol_coords: Optional[List[NDArray]] = None
):
    """construct function to compute du_dx w.r.t. host coords, given fixed mols and box"""

    assert host_config.box.shape == (3, 3)

    # openmm host_system -> timemachine host_bps
    host_bps, _ = openmm_deserializer.deserialize_system(host_config.omm_system, cutoff=1.2)

    # construct appropriate topology from (mols, ff)
    if len(mols) == 1:
        top = topology.BaseTopology(mols[0], ff)
    elif len(mols) == 2:
        top = topology.DualTopology(mols[0], mols[1], ff)
    else:
        raise ValueError("mols must be length 1 or 2")

    hgt = topology.HostGuestTopology(host_bps, top, host_config.num_water_atoms)

    # bound impls of potentials @ lam=0 (fully coupled) endstate
    potentials, params = parameterize_system(hgt, ff, 0.0)
    gpu_impl = summed_potential_bound_impl_from_potentials_and_params(potentials, params)

    # read conformers from mol_coords if given, or each mol's conf0 otherwise
    conf_list = [np.array(host_config.conf)]

    if mol_coords is not None:
        for mc in mol_coords:
            conf_list.append(mc)
    else:
        for mol in mols:
            conf_list.append(get_romol_conf(mol))

    # check conf_list consistent with mols
    assert len(conf_list[1:]) == len(mols)
    for conf, mol in zip(conf_list[1:], mols):
        assert conf.shape == (mol.GetNumAtoms(), 3)

    combined_coords = np.concatenate(conf_list)

    # wrap gpu_impl, partially applying box, mol coords
    num_host_atoms = host_config.conf.shape[0]

    def du_dx_host_fxn(x_host):
        x = np.array(combined_coords)
        x[:num_host_atoms] = x_host

        du_dx, _ = gpu_impl.execute(x, host_config.box, compute_u=False)
        du_dx_host = du_dx[:num_host_atoms]
        return du_dx_host

    return du_dx_host_fxn


def equilibrate_host_barker(
    mols: List[Chem.Mol],
    host_config: HostConfig,
    ff: Forcefield,
    mol_coords: Optional[List[NDArray]] = None,
    temperature: float = DEFAULT_TEMP,
    proposal_stddev: float = 0.0001,
    n_steps: int = 1000,
    seed: Optional[int] = None,
) -> NDArray:
    """Possible alternative to minimize_host_4d, for purposes of clash resolution and initial pre-equilibration

    Notes
    -----
    * Applies a robust proposal targeting lam = 0, and omits Metropolis correction
        * For sufficiently small proposal_stddev, can be expected to sample from approximately the correct distribution
        * Not expected to outperform BAOAB in terms of improved sampling efficiency or reduced sampling bias
        * Possible advantage: robustness w.r.t. clashy initialization
    * Can make progress even when |force| = +inf
    * At proposal_stddev = 0.0001 nm:
        * appears to resolve steric clashes
        * appears stable even with Metropolis correction omitted
    * Can be run as an approximate minimizer by setting temperature == 0.0
    """

    assert 0 < proposal_stddev <= 0.0001, "not tested with Metropolis correction omitted for larger proposal_stddevs"

    du_dx_host_fxn = make_host_du_dx_fxn(mols, host_config, ff, mol_coords)
    grad_log_q = lambda x_host: -du_dx_host_fxn(x_host) / (BOLTZ * temperature)

    # TODO: if needed, revisit choice to omit Metropolis correction
    barker_prop = BarkerProposal(grad_log_q, proposal_stddev, seed=seed)

    x_host = np.array(host_config.conf)

    for t in range(n_steps):
        x_host = barker_prop.sample(x_host)

    final_forces = -du_dx_host_fxn(x_host)
    check_force_norm(final_forces)

    return x_host


def equilibrate_host(
    mol: Chem.Mol,
    host_config: HostConfig,
    temperature: float,
    pressure: float,
    ff: Forcefield,
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

    host_config: HostConfig
        Represents the host system.

    temperature: float
        Temperature at which to run the simulation. Units of kelvins.

    pressure: float
        Pressure at which to run the simulation. Units of bars.

    ff: ff.Forcefield
        Wrapper class around a list of handlers.

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
    host_bps, host_masses = openmm_deserializer.deserialize_system(host_config.omm_system, cutoff=1.2)

    min_host_coords = minimize_host_4d([mol], host_config, ff)

    ligand_masses = get_mol_masses(mol)
    ligand_coords = get_romol_conf(mol)

    combined_masses = np.concatenate([host_masses, ligand_masses])
    combined_coords = np.concatenate([min_host_coords, ligand_coords])

    top = topology.BaseTopology(mol, ff)
    hgt = topology.HostGuestTopology(host_bps, top, host_config.num_water_atoms)

    # setup the parameter handlers for the ligand
    potentials, params = parameterize_system(hgt, ff, 1.0)

    x0 = combined_coords

    # Re-minimize with the mol being flexible
    u_impl = summed_potential_bound_impl_from_potentials_and_params(potentials, params)  # lambda=1
    x0 = fire_minimize(x0, [u_impl], host_config.box, 50)
    v0 = np.zeros_like(x0)

    dt = 2.5e-3
    friction = 1.0

    if seed is None:
        seed = np.random.randint(np.iinfo(np.int32).max)

    hb_potential = next(p for p in potentials if isinstance(p, HarmonicBond))
    bond_list = get_bond_list(hb_potential)
    combined_masses = model_utils.apply_hmr(combined_masses, bond_list)

    integrator = LangevinIntegrator(temperature, dt, friction, combined_masses, seed).impl()

    group_indices = get_group_indices(bond_list, len(combined_masses))

    barostat_interval = 5
    _, params = parameterize_system(hgt, ff, 0.0)  # lambda=0
    u_impl.set_params(flatten_params(params))
    barostat = MonteCarloBarostat(
        x0.shape[0],
        pressure,
        temperature,
        group_indices,
        barostat_interval,
        seed,
    ).impl([u_impl])

    # context components: positions, velocities, box, integrator, energy fxns
    ctxt = custom_ops.Context(x0, v0, host_config.box, integrator, [u_impl], movers=[barostat])

    xs, boxes = ctxt.multiple_steps(n_steps)
    assert len(xs) == 1
    assert len(xs) == len(boxes)
    return xs[-1], boxes[-1]


def get_val_and_grad_fn(bps: Iterable[BoundPotential], box: NDArray, precision=np.float32):
    """
    Convert impls, box into a function that only takes in coords.

    Parameters
    ----------
    bps: List of BoundPotentials

    box: np.array (3,3)

    Returns
    -------
    Energy function with gradient
        f: R^(Nx3) -> (R^1, R^Nx3)
    """
    summed_pot = SummedPotential([bp.potential for bp in bps], [bp.params for bp in bps])
    params = np.concatenate([bp.params.reshape(-1) for bp in bps])
    impl = summed_pot.to_gpu(precision).bind(params).bound_impl

    def val_and_grad_fn(coords):
        g_bp, u_bp = impl.execute(coords, box)
        return u_bp, g_bp

    return val_and_grad_fn


def local_minimize(
    x0: NDArray, val_and_grad_fn, local_idxs, verbose: bool = True, assert_energy_decreased: bool = True
):
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
        if not np.isfinite(u_full) or u_0 - u_full > guard_threshold:
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

    U_final, grad_final = val_and_grad_fn_bfgs(x_local_final_flat)
    forces = -grad_final.reshape(x_local_shape)
    per_atom_force_norms = np.linalg.norm(forces, axis=1)

    if verbose:
        print(f"U(x_final) = {U_final:.3f}")
        # diagnose worst atom
        argmax_local = np.argmax(per_atom_force_norms)
        worst_atom_idx = local_idxs[argmax_local]
        print(f"atom with highest force norm after minimization: {worst_atom_idx}")
        print(f"force(x_final)[{worst_atom_idx}] = {forces[argmax_local]}")
        print("-" * 70)

    # note that this over the local atoms only, as this function is not concerned
    check_force_norm(forces)

    x_final = x0.copy()
    x_final[local_idxs] = x_local_final

    if assert_energy_decreased:
        assert U_final < u_0, f"u_0: {u_0:.3f}, u_f: {U_final:.3f}"
    elif U_final >= u_0:
        warnings.warn(f"WARNING: Energy did not decrease: u_0: {u_0:.3f}, u_f: {U_final:.3f}", MinimizationWarning)

    return x_final


def replace_conformer_with_minimized(mol: Chem.rdchem.Mol, ff: Forcefield):
    """Replace the first conformer of the given mol with a conformer minimized with respect to the given forcefield.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Input mol. Must have at least one conformer.

    ff : Forcefield
        Forcefield to use in energy minimization
    """
    top = topology.BaseTopology(mol, ff)
    system = top.setup_end_state()
    val_and_grad_fn = jax.value_and_grad(system.get_U_fn())
    xs = get_romol_conf(mol)
    all_idxs = np.arange(mol.GetNumAtoms())
    xs_opt = local_minimize(xs, val_and_grad_fn, all_idxs, verbose=False)
    set_romol_conf(mol, xs_opt)
