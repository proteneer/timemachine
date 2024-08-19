import warnings
from typing import Callable, Iterable, List, Optional, Tuple

import jax
import numpy as np
import scipy.optimize
from numpy.typing import NDArray
from rdkit import Chem

from timemachine.constants import BOLTZ, DEFAULT_PRESSURE, DEFAULT_TEMP, MAX_FORCE_NORM
from timemachine.fe import topology
from timemachine.fe.free_energy import HostConfig
from timemachine.fe.utils import get_romol_conf, set_romol_conf
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


def fire_minimize(
    x0: NDArray,
    du_dx_fxn: Callable[[NDArray], NDArray],
    n_steps: int,
) -> NDArray:
    """
    Minimize coordinates using the FIRE algorithm

    Parameters
    ----------
    coords: np.ndarray
        N x 3 coordinates. units of nanometers.

    du_dx_fxn: Function that given coordinates returns the du_dx

    n_steps: int
        Number of steps

    Returns
    -------
    np.ndarray
        Minimized coords.

    """

    def force(coords):
        forces = -du_dx_fxn(coords)
        return forces

    def shift(d, dr):
        return d + dr

    init, f = fire_descent(force, shift)
    opt_state = init(x0)
    for _ in range(n_steps):
        opt_state = f(opt_state)
    return np.asarray(opt_state.position)


def pre_equilibrate_host(
    mols: List[Chem.Mol],
    host_config: HostConfig,
    ff: Forcefield,
    mol_coords: Optional[List[NDArray]] = None,
    minimizer_steps_per_window: int = 500,
    minimizer_windows: int = 2,
    minimizer_max_lambda: float = 0.1,
    equilibration_steps: int = 1000,
    pressure: float = DEFAULT_PRESSURE,
    temperature: float = DEFAULT_TEMP,
    barostat_interval: int = 5,
    seed: int = 2024,
) -> Tuple[NDArray, NDArray]:
    """pre_equilibrate_host is a utility function that performs minimization than some amount of equilibration.

    The intention of this function is to resolve any potential clashes in the system, then to equilibrate the box size, all while
    keeping the ligand fixed. This helps set up up simulations where the local region around the ligand is optimized over a set of windows.

    Parameters
    ----------
    mols: list of Chem.Mol
        Ligands to be inserted. This must be of length 1 or 2 for now.

    host_config: HostConfig
        Represents the host system.

    ff: ff.Forcefield
        Wrapper class around a list of handlers

    mol_coords: list of np.ndarray, optional
        Pre-specify a list of mol coords. Else use the mol.GetConformer(0)

    minimizer_steps_per_window: integer
        Number of steps to run each window of the FIRE minimizer with

    minimizer_windows: integer
        Number of windows to run FIRE minimizer over

    minimizer_max_lambda: float
        The largest lambda value to run the minimizer with. Refer to docstring of fire_minimize_host for more
        details.

    equilibration_steps: integer
        Number of steps to run MD with

    pressure: float
        in bar (used by barostat)

    temperature: float
        in kelvin (used by integrator and barostat)

    barostat_interval: integer
        # of MD steps between barostat moves

    seed: integer
        integrator uses `seed`, barostat uses `seed+1`

    Returns
    -------
    2-tuple of host coordinates and box
        the minimized host_coords and the new box

    """
    box = host_config.box
    assert box.shape == (3, 3)

    minimized_host_coords = fire_minimize_host(
        mols,
        host_config,
        ff,
        mol_coords=mol_coords,
        n_windows=minimizer_windows,
        n_steps_per_window=minimizer_steps_per_window,
        max_lambda=minimizer_max_lambda,
    )

    host_bps, host_masses = openmm_deserializer.deserialize_system(host_config.omm_system, cutoff=1.2)

    num_host_atoms = host_config.conf.shape[0]

    if len(mols) == 1:
        top = topology.BaseTopology(mols[0], ff)
    elif len(mols) == 2:
        top = topology.DualTopologyMinimization(mols[0], mols[1], ff)
    else:
        raise ValueError("mols must be length 1 or 2")

    mass_list = [np.array(host_masses)]
    conf_list = [minimized_host_coords]
    for mol in mols:
        # Set ligand masses to inf to ensure ligands don't move
        mass_list.append(np.ones(mol.GetNumAtoms()) * np.inf)

    if mol_coords is None:
        mol_coords = [get_romol_conf(mol) for mol in mols]
    for mc in mol_coords:
        conf_list.append(mc)

    combined_masses = np.concatenate(mass_list)
    combined_coords = np.concatenate(conf_list)

    hgt = topology.HostGuestTopology(host_bps, top, host_config.num_water_atoms)

    dt = 1.5e-3
    friction = 1.0

    intg = LangevinIntegrator(temperature, dt, friction, combined_masses, seed).impl()

    x0 = combined_coords
    v0 = np.zeros_like(x0)

    num_host_atoms = minimized_host_coords.shape[0]

    potentials, params = parameterize_system(hgt, ff, 0.0)
    bond_pot = next(pot for pot in potentials if isinstance(pot, HarmonicBond))
    group_idxs = get_group_indices(get_bond_list(bond_pot), x0.shape[0])
    # Disallow the barostat from scaling the ligand coords, scale all of the other molecules to
    # reduce 'air bubbles' within the system. Less efficient than scaling the entire system, but
    # don't want to adjust the ligand coordinates at all.
    non_ligand_group_idxs = [group for group in group_idxs if np.all(group < num_host_atoms)]

    u_impl = summed_potential_bound_impl_from_potentials_and_params(potentials, params)
    bound_impls = [u_impl]

    baro = MonteCarloBarostat(
        x0.shape[0],
        pressure,
        temperature,
        non_ligand_group_idxs,
        barostat_interval,
        seed + 1,
    )
    baro_impl = baro.impl(bound_impls)

    ctxt = custom_ops.Context(x0, v0, box, intg, bound_impls, movers=[baro_impl])
    xs, boxes = ctxt.multiple_steps(equilibration_steps)
    x = xs[-1]
    box = boxes[-1]

    assert np.all(x[num_host_atoms:] == np.concatenate(mol_coords)), "Ligand atoms unexpectedly moved"

    # Only evaluate the host forces, the mols may be strained at this stage. No change is made to the mols
    # which means evaluating the mols forces may trigger spurious failures
    for impl in bound_impls:
        du_dx, _ = impl.execute(x, box, compute_u=False)
        check_force_norm(-du_dx[:num_host_atoms])

    return x[:num_host_atoms], box


def fire_minimize_host(
    mols: List[Chem.Mol],
    host_config: HostConfig,
    ff: Forcefield,
    mol_coords: Optional[List[NDArray]] = None,
    n_steps_per_window: int = 500,
    max_lambda: float = 0.1,
    n_windows: int = 2,
) -> NDArray:
    """
    Minimize a host system using the Fire minimizer.

    The ligand coordinates are fixed during minimization, and only host_coords are minimized.

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

    n_steps_per_window: integer
        Number of steps to run the FIRE minimizer with at each window

    max_lambda: float
        The largest lambda value to run a window at. If the value is 1.0 the mols will be
        completely decoupled. Between lambda 1.0 and 0.0 the 4D coordinate of the mols are linearly
        interpolated from nonbonded cutoff to 0.0, no others parameters are modified.

    n_windows: integer
        The number of windows to linearly interpolate between the max_lambda value and 0.0.

    Returns
    -------
    np.ndarray
        the minimized host_coords.

    """

    assert 1.0 >= max_lambda > 0.0, "Max lambda must be greater than 0.0 and less than or equal to 1.0"
    x_host = np.asarray(host_config.conf)

    for lamb in np.linspace(max_lambda, 0.0, n_windows):
        du_dx_fxn = make_host_du_dx_fxn(mols, host_config, ff, mol_coords=mol_coords, lamb=lamb)
        x_host = fire_minimize(x_host, du_dx_fxn, n_steps_per_window)

    du_dx = du_dx_fxn(x_host)
    check_force_norm(-du_dx)

    return x_host


def make_host_du_dx_fxn(
    mols: List[Chem.Mol],
    host_config: HostConfig,
    ff: Forcefield,
    mol_coords: Optional[List[NDArray]] = None,
    lamb: float = 0.0,
):
    """construct function to compute du_dx w.r.t. host coords, given fixed mols and box"""

    assert host_config.box.shape == (3, 3)

    # openmm host_system -> timemachine host_bps
    host_bps, _ = openmm_deserializer.deserialize_system(host_config.omm_system, cutoff=1.2)

    # construct appropriate topology from (mols, ff)
    if len(mols) == 1:
        top = topology.BaseTopology(mols[0], ff)
    elif len(mols) == 2:
        top = topology.DualTopologyMinimization(mols[0], mols[1], ff)
    else:
        raise ValueError("mols must be length 1 or 2")

    hgt = topology.HostGuestTopology(host_bps, top, host_config.num_water_atoms)

    # read conformers from mol_coords if given, or each mol's conf0 otherwise
    conf_list = [np.asarray(host_config.conf)]

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

    # bound impls of potentials @ lam=0 (fully coupled) endstate
    potentials, params = parameterize_system(hgt, ff, lamb)
    gpu_impl = summed_potential_bound_impl_from_potentials_and_params(potentials, params)

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
    """Possible alternative to fire_minimize_host, for purposes of clash resolution and initial pre-equilibration

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

    def val_and_grad_fn_local(x_local):
        x_prime = x0.copy()
        x_prime[local_idxs] = x_local
        u_full, grad_full = val_and_grad_fn(x_prime)
        # The GPU Potentials can return NaN if value would have overflowed in uint64
        if np.isnan(u_full):
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
