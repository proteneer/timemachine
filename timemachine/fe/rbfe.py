import pickle
import traceback
import warnings
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from openmm import app
from rdkit import Chem

from timemachine.constants import DEFAULT_ATOM_MAPPING_KWARGS, DEFAULT_PRESSURE, DEFAULT_TEMP
from timemachine.fe import atom_mapping, model_utils
from timemachine.fe.free_energy import (
    HostConfig,
    InitialState,
    MDParams,
    SimulationResult,
    make_pair_bar_plots,
    run_sims_bisection,
    run_sims_hrex,
    run_sims_sequential,
)
from timemachine.fe.single_topology import SingleTopology
from timemachine.fe.system import VacuumSystem, convert_omm_system
from timemachine.fe.utils import bytes_to_id, get_mol_name, get_romol_conf
from timemachine.ff import Forcefield
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat
from timemachine.md import builders, minimizer
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.parallel.client import AbstractClient, AbstractFileClient, CUDAPoolClient, FileClient
from timemachine.potentials import BoundPotential, jax_utils

DEFAULT_NUM_WINDOWS = 30

# the constant is arbitrary, but see
# https://github.com/proteneer/timemachine/commit/e1f7328f01f427534d8744aab6027338e116ad09
MAX_SEED_VALUE = 10000

DEFAULT_MD_PARAMS = MDParams(n_frames=1000, n_eq_steps=10_000, steps_per_frame=400, seed=2023)


@dataclass
class Host:
    system: VacuumSystem
    physical_masses: List[float]
    conf: NDArray
    box: NDArray
    num_water_atoms: int


def setup_in_vacuum(st: SingleTopology, ligand_conf, lamb):
    """Prepare potentials, initial coords, large 10x10x10nm box, and HMR masses"""

    system = st.setup_intermediate_state(lamb)
    hmr_masses = np.array(st.combine_masses(use_hmr=True))

    potentials = system.get_U_fns()
    baro = None

    x0 = ligand_conf
    box0 = np.eye(3, dtype=np.float64) * 10  # make a large 10x10x10nm box

    return x0, box0, hmr_masses, potentials, baro


def setup_in_env(
    st: SingleTopology,
    host: Host,
    ligand_conf: NDArray,
    lamb: float,
    temperature: float,
    run_seed: int,
):
    """Prepare potentials, concatenate environment and ligand coords, apply HMR, and construct barostat"""
    system = st.combine_with_host(host.system, lamb, host.num_water_atoms)
    host_hmr_masses = model_utils.apply_hmr(host.physical_masses, host.system.bond.potential.idxs)
    hmr_masses = np.concatenate([host_hmr_masses, st.combine_masses(use_hmr=True)])

    potentials = system.get_U_fns()
    group_idxs = get_group_indices(get_bond_list(system.bond.potential), len(hmr_masses))
    baro = MonteCarloBarostat(len(hmr_masses), DEFAULT_PRESSURE, temperature, group_idxs, 15, run_seed + 1)

    x0 = np.concatenate([host.conf, ligand_conf])

    return x0, hmr_masses, potentials, baro


def assert_all_states_have_same_masses(initial_states: List[InitialState]):
    """
    hmr masses should be identical throughout the lambda schedule
    bond idxs should be the same at the two end-states, note that a possible corner
    case with bond breaking may seem to be problematic:

    0 1 2    0 1 2
    C-O-C -> C.H-C

    but this isn't an issue, since hydrogens will only ever be terminal atoms
    and core hydrogens that are mapped to heavy atoms will take the mass of the
    heavy atom (thereby not triggering the mass repartitioning to begin with).

    but it's reasonable to be skeptical, so we also assert consistency through the lambda
    schedule as an extra sanity check.
    """

    masses = np.array([s.integrator.masses for s in initial_states])
    deviation_among_windows = masses.std(0)
    np.testing.assert_array_almost_equal(deviation_among_windows, 0, err_msg="masses assumed constant w.r.t. lambda")


def setup_initial_state(
    st: SingleTopology,
    lamb: float,
    host: Optional[Host],
    temperature: float,
    seed: int,
) -> InitialState:
    conf_a = get_romol_conf(st.mol_a)
    conf_b = get_romol_conf(st.mol_b)

    ligand_conf = st.combine_confs(conf_a, conf_b, lamb)
    # use a different seed to initialize every window,
    # but in a way that should be symmetric for
    # A -> B vs. B -> A edge definitions
    init_seed = int(seed + bytes_to_id(ligand_conf.tobytes())) % MAX_SEED_VALUE
    if host:
        x0, hmr_masses, potentials, baro = setup_in_env(st, host, ligand_conf, lamb, temperature, init_seed)
        box0 = host.box
    else:
        x0, box0, hmr_masses, potentials, baro = setup_in_vacuum(st, ligand_conf, lamb)

    # provide a different run_seed for every lambda window,
    # but in a way that should be symmetric for
    # A -> B vs. B -> A edge definitions
    run_seed = (
        int(seed + bytes_to_id(bytes().join([np.array(p.params).tobytes() for p in potentials]))) % MAX_SEED_VALUE
    )

    # initialize velocities
    v0 = np.zeros_like(x0)  # tbd resample from Maxwell-boltzman?

    # determine ligand idxs
    num_ligand_atoms = len(ligand_conf)
    num_total_atoms = len(x0)
    ligand_idxs = np.arange(num_total_atoms - num_ligand_atoms, num_total_atoms)

    # initialize Langevin integrator
    dt = 2.5e-3
    friction = 1.0
    intg = LangevinIntegrator(temperature, dt, friction, hmr_masses, run_seed)

    return InitialState(potentials, intg, baro, x0, v0, box0, lamb, ligand_idxs)


def setup_optimized_host(st: SingleTopology, config: HostConfig) -> Host:
    system, masses = convert_omm_system(config.omm_system)
    conf = minimizer.minimize_host_4d(
        [st.mol_a, st.mol_b],
        config,
        st.ff,
    )
    return Host(system, masses, conf, config.box, config.num_water_atoms)


def setup_initial_states(
    st: SingleTopology,
    host: Optional[Host],
    temperature: float,
    lambda_schedule: Union[NDArray, Sequence[float]],
    seed: int,
    min_cutoff: Optional[float] = 0.7,
) -> List[InitialState]:
    """
    Given a sequence of lambda values, return a list of initial states.

    The InitialState objects can be used to recover a bitwise-identical simulation for debugging.

    Assumes lambda schedule is a monotonically increasing sequence in the closed interval [0, 1].

    Parameters
    ----------
    st: SingleTopology
        A single topology object

    host: Host or None
        Pre-optimized host configuration, generated using `setup_optimized_host`. If None, return vacuum states.

    temperature: float
        Temperature to run the simulation at.

    lambda_schedule: list of float of length K
        Lambda schedule.

    seed: int
        Random number seed

    min_cutoff: float, optional
        Throw error if any atom moves more than this distance (nm) after minimization

    Returns
    -------
    list of InitialState
        Initial state for each value of lambda.
    """

    # check that the lambda schedule is monotonically increasing.
    assert np.all(np.diff(lambda_schedule) > 0)

    initial_states = [setup_initial_state(st, lamb, host, temperature, seed) for lamb in lambda_schedule]

    # minimize ligand and environment atoms within min_cutoff of the ligand
    # optimization introduces dependencies among states with lam < 0.5, and among states with lam >= 0.5
    optimized_x0s = optimize_coordinates(initial_states, min_cutoff=min_cutoff)

    # update initial states in-place
    for state, x0 in zip(initial_states, optimized_x0s):
        state.x0 = x0

    # perform any concluding sanity-checks
    assert_all_states_have_same_masses(initial_states)

    return initial_states


def setup_optimized_initial_state(
    st: SingleTopology,
    lamb: float,
    host: Optional[Host],
    optimized_initial_states: Sequence[InitialState],
    temperature: float,
    seed: int,
) -> InitialState:

    # Use pre-optimized initial state with the closest value of lambda as a starting point for optimization.

    # NOTE: The current approach for generating optimized conformations in `optimize_coordinates` creates a
    # discontinuity at lambda=0.5. Ensure that we pick a pre-optimized state on the same side of 0.5 as `lamb`:
    states_subset = [s for s in optimized_initial_states if (s.lamb <= 0.5) == (lamb <= 0.5)]
    nearest_optimized = min(states_subset, key=lambda s: abs(lamb - s.lamb))

    if lamb == nearest_optimized.lamb:
        return nearest_optimized
    else:
        initial_state = setup_initial_state(st, lamb, host, temperature, seed)
        free_idxs = get_free_idxs(nearest_optimized)
        initial_state.x0 = optimize_coords_state(
            initial_state.potentials,
            nearest_optimized.x0,
            initial_state.box0,
            free_idxs,
            # assertion can lead to spurious errors when new state is close to an existing one
            assert_energy_decreased=False,
        )
        return initial_state


def optimize_coords_state(
    potentials: Iterable[BoundPotential],
    x0: NDArray,
    box: NDArray,
    free_idxs: List[int],
    assert_energy_decreased: bool,
) -> NDArray:
    val_and_grad_fn = minimizer.get_val_and_grad_fn(potentials, box)
    assert np.all(np.isfinite(x0)), "Initial coordinates contain nan or inf"
    x_opt = minimizer.local_minimize(x0, val_and_grad_fn, free_idxs, assert_energy_decreased=assert_energy_decreased)
    assert np.all(np.isfinite(x_opt)), "Minimization resulted in a nan"
    return x_opt


def get_free_idxs(initial_state: InitialState, cutoff: float = 0.5) -> List[int]:
    """Select particles within cutoff of ligand"""
    x = initial_state.x0
    x_lig = x[initial_state.ligand_idxs]
    box = initial_state.box0
    free_idxs = jax_utils.idxs_within_cutoff(x, x_lig, box, cutoff=cutoff).tolist()
    return free_idxs


def _optimize_coords_along_states(initial_states: List[InitialState]) -> List[NDArray]:
    # use the end-state to define the optimization settings
    end_state = initial_states[0]
    x_opt = end_state.x0

    x_traj = []
    for idx, initial_state in enumerate(initial_states):
        print(f"Optimizing initial state at λ={initial_state.lamb}")
        free_idxs = get_free_idxs(initial_state)
        x_opt = optimize_coords_state(
            initial_state.potentials, x_opt, initial_state.box0, free_idxs, assert_energy_decreased=idx == 0
        )
        x_traj.append(x_opt)

    return x_traj


def optimize_coordinates(initial_states, min_cutoff=0.7) -> List[NDArray]:
    """
    Optimize geometries of the initial states.

    Parameters
    ----------
    initial_states: list of InitialState

    min_cutoff: float, optional
        Throw error if any atom moves more than this distance (nm) after minimization

    Returns
    -------
    list of np.array
        Optimized coordinates

    """
    all_xs = []
    lambda_schedule = np.array([s.lamb for s in initial_states])

    # check for monotonic, any subsequence of a monotonic sequence is also monotonic.
    assert np.all(np.diff(lambda_schedule) > 0)

    lhs_initial_states = []
    rhs_initial_states = []

    for state in initial_states:
        if state.lamb < 0.5:
            lhs_initial_states.append(state)
        else:
            rhs_initial_states.append(state)

    # go from lambda 0 -> 0.5
    if len(lhs_initial_states) > 0:
        lhs_xs = _optimize_coords_along_states(lhs_initial_states)
        for xs in lhs_xs:
            all_xs.append(xs)

    # go from lambda 1 -> 0.5 and reverse the coordinate trajectory and lambda schedule
    if len(rhs_initial_states) > 0:
        rhs_xs = _optimize_coords_along_states(rhs_initial_states[::-1])[::-1]
        for xs in rhs_xs:
            all_xs.append(xs)

    # sanity check that no atom has moved more than `min_cutoff` nm away
    for state, coords in zip(initial_states, all_xs):
        displacement_distances = jax_utils.distance_on_pairs(state.x0, coords, box=state.box0)
        if min_cutoff is not None:
            assert (
                displacement_distances < min_cutoff
            ).all(), f"λ = {state.lamb} moved an atom > {min_cutoff*10} Å from initial state during minimization"

    return all_xs


def estimate_relative_free_energy(
    mol_a: Chem.rdchem.Mol,
    mol_b: Chem.rdchem.Mol,
    core: NDArray,
    ff: Forcefield,
    host_config: Optional[HostConfig],
    prefix: str = "",
    lambda_interval: Optional[Tuple[float, float]] = None,
    n_windows: Optional[int] = None,
    keep_idxs: Optional[List[int]] = None,
    md_params: MDParams = DEFAULT_MD_PARAMS,
    min_cutoff: Optional[float] = 0.7,
) -> SimulationResult:
    """
    Estimate relative free energy between mol_a and mol_b via independent simulations with a predetermined lambda
    schedule. Molecules should be aligned to each other and within the host environment.

    Parameters
    ----------
    mol_a: Chem.Mol
        initial molecule

    mol_b: Chem.Mol
        target molecule

    core: list of 2-tuples
        atom_mapping of atoms in mol_a into atoms in mol_b

    ff: Forcefield
        Forcefield to be used for the system

    host_config: HostConfig or None
        Configuration for the host system. If None, then the vacuum leg is run.

    prefix: str
        A prefix to append to figures

    lambda_interval: (float, float) or None, optional
        Minimum and maximum value of lambda for the transformation; typically (0, 1), but sometimes useful to choose
        other values for testing.

    n_windows: int or None, optional
        Number of windows used for interpolating the lambda schedule with additional windows. Defaults to
        `DEFAULT_NUM_WINDOWS` windows.

    keep_idxs: list of int or None, optional
        If None, return only the end-state frames. Otherwise if not None, use only for debugging, and this
        will return the frames corresponding to the idxs of interest.

    md_params: MDParams, optional
        Parameters for the equilibration and production MD. Defaults to :py:const:`timemachine.fe.rbfe.DEFAULT_MD_PARAMS`

    min_cutoff: float, optional
        Throw error if any atom moves more than this distance (nm) after minimization

    Returns
    -------
    SimulationResult
        Collected data from the simulation (see class for storage information). Returned frames and boxes
        are defined by keep_idxs.

    """
    single_topology = SingleTopology(mol_a, mol_b, core, ff)

    lambda_min, lambda_max = lambda_interval or (0.0, 1.0)
    lambda_schedule = np.linspace(lambda_min, lambda_max, n_windows or DEFAULT_NUM_WINDOWS)

    temperature = DEFAULT_TEMP

    host = setup_optimized_host(single_topology, host_config) if host_config else None

    initial_states = setup_initial_states(
        single_topology, host, temperature, lambda_schedule, md_params.seed, min_cutoff=min_cutoff
    )

    if keep_idxs is None:
        keep_idxs = [0, -1]  # keep frames from first and last windows
    assert len(keep_idxs) <= len(lambda_schedule)

    # TODO: rename prefix to postfix, or move to beginning of combined_prefix?
    combined_prefix = get_mol_name(mol_a) + "_" + get_mol_name(mol_b) + "_" + prefix
    try:
        result, stored_frames, stored_boxes = run_sims_sequential(initial_states, md_params, temperature, keep_idxs)
        plots = make_pair_bar_plots(result, temperature, combined_prefix)
        return SimulationResult(result, plots, stored_frames, stored_boxes, md_params, [])
    except Exception as err:
        with open(f"failed_rbfe_result_{combined_prefix}.pkl", "wb") as fh:
            pickle.dump((initial_states, md_params, err), fh)
        raise err


def estimate_relative_free_energy_bisection(
    mol_a: Chem.rdchem.Mol,
    mol_b: Chem.rdchem.Mol,
    core: NDArray,
    ff: Forcefield,
    host_config: Optional[HostConfig],
    md_params: MDParams = DEFAULT_MD_PARAMS,
    prefix: str = "",
    lambda_interval: Optional[Tuple[float, float]] = None,
    n_windows: Optional[int] = None,
    min_overlap: Optional[float] = None,
    keep_idxs: Optional[List[int]] = None,
    min_cutoff: Optional[float] = 0.7,
) -> SimulationResult:
    r"""Estimate relative free energy between mol_a and mol_b via independent simulations with a dynamic lambda schedule
    determined by successively bisecting the lambda interval between the pair of states with the greatest BAR
    :math:`\Delta G` error. Molecules should be aligned to each other and within the host environment.

    Parameters
    ----------
    mol_a: Chem.Mol
        initial molecule

    mol_b: Chem.Mol
        target molecule

    core: list of 2-tuples
        atom_mapping of atoms in mol_a into atoms in mol_b

    ff: Forcefield
        Forcefield to be used for the system

    host_config: HostConfig or None
        Configuration for the host system. If None, then the vacuum leg is run.

    md_params: MDParams, optional
        Parameters for the equilibration and production MD. Defaults to :py:const:`timemachine.fe.rbfe.DEFAULT_MD_PARAMS`

    prefix: str, optional
        A prefix to append to figures

    lambda_interval: (float, float) or None, optional
        Minimum and maximum value of lambda for the transformation; typically (0, 1), but sometimes useful to choose
        other values for testing.

    n_windows: int or None, optional
        Number of windows used for interpolating the lambda schedule with additional windows. Additionally controls the
        number of evenly-spaced lambda windows used for initial conformer optimization. Defaults to
        `DEFAULT_NUM_WINDOWS` windows.

    min_overlap: float or None, optional
        If not None, terminate bisection early when the BAR overlap between all neighboring pairs of states exceeds this
        value. When given, the final number of windows may be less than or equal to n_windows.

    keep_idxs: list of int or None, optional
        If None, return only the end-state frames. Otherwise if not None (typically for debugging), return frames from
        windows corresponding to the specified indices.

    min_cutoff: float or None, optional
        Throw error if any atom moves more than this distance (nm) after minimization

    Returns
    -------
    SimulationResult
        Collected data from the simulation (see class for storage information). Returned frames and boxes
        are defined by keep_idxs.
    """

    if n_windows is None:
        n_windows = DEFAULT_NUM_WINDOWS
    assert n_windows >= 2

    if keep_idxs is None:
        keep_idxs = [0, -1]  # keep frames from first and last windows

    assert len(set(keep_idxs)) == len(keep_idxs)
    assert len(keep_idxs) <= n_windows

    single_topology = SingleTopology(mol_a, mol_b, core, ff)

    lambda_min, lambda_max = lambda_interval or (0.0, 1.0)
    lambda_grid = np.linspace(lambda_min, lambda_max, n_windows)

    temperature = DEFAULT_TEMP

    host = setup_optimized_host(single_topology, host_config) if host_config else None

    initial_states = setup_initial_states(
        single_topology, host, temperature, lambda_grid, md_params.seed, min_cutoff=min_cutoff
    )

    make_optimized_initial_state = partial(
        setup_optimized_initial_state,
        single_topology,
        host=host,
        optimized_initial_states=initial_states,
        temperature=temperature,
        seed=md_params.seed,
    )

    # TODO: rename prefix to postfix, or move to beginning of combined_prefix?
    combined_prefix = get_mol_name(mol_a) + "_" + get_mol_name(mol_b) + "_" + prefix

    try:
        results, frames, boxes = run_sims_bisection(
            [lambda_min, lambda_max],
            make_optimized_initial_state,
            md_params,
            n_bisections=len(lambda_grid) - 2,
            temperature=temperature,
            min_overlap=min_overlap,
        )

        final_result = results[-1]

        plots = make_pair_bar_plots(final_result, temperature, combined_prefix)

        assert len(frames) == len(boxes) == len(results) + 1
        stored_frames = []
        stored_boxes = []
        for i in keep_idxs:
            try:
                stored_frames.append(frames[i])
                stored_boxes.append(boxes[i])
            except IndexError:
                warnings.warn(f"Invalid index in keep_idxs: {i}. Bisection terminated with only {len(frames)} windows.")

        return SimulationResult(
            final_result,
            plots,
            stored_frames,
            stored_boxes,
            md_params,
            results,
        )

    except Exception as err:
        with open(f"failed_rbfe_result_{combined_prefix}.pkl", "wb") as fh:
            pickle.dump((md_params, err), fh)
        raise err


def estimate_relative_free_energy_bisection_hrex(
    mol_a: Chem.rdchem.Mol,
    mol_b: Chem.rdchem.Mol,
    core: NDArray,
    ff: Forcefield,
    host_config: Optional[HostConfig],
    md_params: MDParams = DEFAULT_MD_PARAMS,
    prefix: str = "",
    lambda_interval: Optional[Tuple[float, float]] = None,
    n_windows: Optional[int] = None,
    min_overlap: Optional[float] = None,
    keep_idxs: Optional[List[int]] = None,
    min_cutoff: Optional[float] = 0.7,
    n_frames_bisection: int = 100,
    n_frames_per_iter: int = 10,
) -> SimulationResult:
    """
    Estimate relative free energy between mol_a and mol_b using Hamiltonian Replica EXchange (HREX) sampling of a
    sequence of intermediate states determined by bisection. Molecules should be aligned to each other and within the
    host environment.

    Parameters
    ----------
    mol_a: Chem.Mol
        initial molecule

    mol_b: Chem.Mol
        target molecule

    core: list of 2-tuples
        atom_mapping of atoms in mol_a into atoms in mol_b

    ff: Forcefield
        Forcefield to be used for the system

    host_config: HostConfig or None
        Configuration for the host system. If None, then the vacuum leg is run.

    md_params: MDParams, optional
        Parameters for the equilibration and production MD. Defaults to :py:const:`timemachine.fe.rbfe.DEFAULT_MD_PARAMS`

    prefix: str, optional
        A prefix to append to figures

    lambda_interval: (float, float) or None, optional
        Minimum and maximum value of lambda for the transformation; typically (0, 1), but sometimes useful to choose
        other values for testing.

    n_windows: int or None, optional
        Number of windows used for interpolating the lambda schedule with additional windows. Defaults to
        `DEFAULT_NUM_WINDOWS` windows.

    min_overlap: float or None, optional
        If not None, terminate bisection early when the BAR overlap between all neighboring pairs of states exceeds this
        value. When given, the final number of windows may be less than or equal to n_windows.

    keep_idxs: list of int or None, optional
        If None, return only the end-state frames. Otherwise if not None, use only for debugging, and this
        will return the frames corresponding to the idxs of interest.

    min_cutoff: float or None, optional
        Throw error if any atom moves more than this distance (nm) after minimization

    n_frames_bisection: int or None, optional
        Number of frames to sample using MD during the initial bisection phase used to determine lambda spacing

    n_frames_per_iter: int or None, optional
        Number of frames to sample using MD per HREX iteration

    Returns
    -------
    SimulationResult
        Collected data from the simulation (see class for storage information). Returned frames and boxes
        are defined by keep_idxs.
    """

    if n_windows is None:
        n_windows = DEFAULT_NUM_WINDOWS
    assert n_windows >= 2

    if keep_idxs is None:
        keep_idxs = [0, -1]  # keep frames from first and last windows

    assert len(set(keep_idxs)) == len(keep_idxs)
    assert len(keep_idxs) <= n_windows

    single_topology = SingleTopology(mol_a, mol_b, core, ff)

    lambda_min, lambda_max = lambda_interval or (0.0, 1.0)
    lambda_grid = np.linspace(lambda_min, lambda_max, n_windows)

    temperature = DEFAULT_TEMP

    host = setup_optimized_host(single_topology, host_config) if host_config else None

    initial_states = setup_initial_states(
        single_topology, host, temperature, lambda_grid, md_params.seed, min_cutoff=min_cutoff
    )

    make_optimized_initial_state = partial(
        setup_optimized_initial_state,
        single_topology,
        host=host,
        optimized_initial_states=initial_states,
        temperature=temperature,
        seed=md_params.seed,
    )

    # TODO: rename prefix to postfix, or move to beginning of combined_prefix?
    combined_prefix = get_mol_name(mol_a) + "_" + get_mol_name(mol_b) + "_" + prefix

    try:
        # First phase: bisection to determine lambda spacing
        results_bisection, _, _ = run_sims_bisection(
            [lambda_min, lambda_max],
            make_optimized_initial_state,
            replace(md_params, n_frames=n_frames_bisection),  # TODO: clean up
            n_bisections=len(lambda_grid) - 2,
            temperature=temperature,
            min_overlap=min_overlap,
        )
        result_bisection = results_bisection[-1]

        # Second phase: sample initial states determined by bisection using HREX
        initial_states_hrex = result_bisection.initial_states
        pair_bar_result, frames, boxes, diagnostics = run_sims_hrex(
            initial_states_hrex, md_params, n_frames_per_iter=n_frames_per_iter, temperature=temperature
        )

        plots = make_pair_bar_plots(pair_bar_result, temperature, combined_prefix)

        stored_frames = []
        stored_boxes = []
        for i in keep_idxs:
            try:
                stored_frames.append(frames[i])
                stored_boxes.append(boxes[i])
            except IndexError:
                warnings.warn(f"Invalid index in keep_idxs: {i}. Bisection terminated with only {len(frames)} windows.")

        return SimulationResult(
            pair_bar_result, plots, stored_frames, stored_boxes, md_params, results_bisection, diagnostics
        )

    except Exception as err:
        with open(f"failed_rbfe_result_{combined_prefix}.pkl", "wb") as fh:
            pickle.dump((md_params, err), fh)
        raise err


def run_vacuum(
    mol_a: Chem.rdchem.Mol,
    mol_b: Chem.rdchem.Mol,
    core: NDArray,
    forcefield: Forcefield,
    _,
    md_params: MDParams = DEFAULT_MD_PARAMS,
    n_windows: Optional[int] = None,
    min_overlap: Optional[float] = None,
    keep_idxs: Optional[List[int]] = None,
    min_cutoff: Optional[float] = None,
):
    if md_params is not None and md_params.local_steps > 0:
        md_params = replace(md_params, local_steps=0)
        warnings.warn("Vacuum simulations don't support local steps, will use all global steps")
    # min_cutoff defaults to None since there is no environment to prevent conformational changes in the ligand
    return estimate_relative_free_energy_bisection(
        mol_a,
        mol_b,
        core,
        forcefield,
        md_params=md_params,
        host_config=None,
        prefix="vacuum",
        n_windows=n_windows,
        min_overlap=min_overlap,
        keep_idxs=keep_idxs,
        min_cutoff=min_cutoff,
    )


def run_solvent(
    mol_a: Chem.rdchem.Mol,
    mol_b: Chem.rdchem.Mol,
    core: NDArray,
    forcefield: Forcefield,
    _,
    md_params: MDParams = DEFAULT_MD_PARAMS,
    n_windows: Optional[int] = None,
    min_overlap: Optional[float] = None,
    keep_idxs: Optional[List[int]] = None,
    min_cutoff: Optional[float] = 0.7,
):
    box_width = 4.0
    solvent_sys, solvent_conf, solvent_box, solvent_top = builders.build_water_system(box_width, forcefield.water_ff)
    solvent_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes, deboggle later
    solvent_host_config = HostConfig(solvent_sys, solvent_conf, solvent_box, solvent_conf.shape[0])
    solvent_res = estimate_relative_free_energy_bisection(
        mol_a,
        mol_b,
        core,
        forcefield,
        solvent_host_config,
        md_params=md_params,
        prefix="solvent",
        n_windows=n_windows,
        min_overlap=min_overlap,
        keep_idxs=keep_idxs,
        min_cutoff=min_cutoff,
    )
    return solvent_res, solvent_top, solvent_host_config


def run_complex(
    mol_a: Chem.rdchem.Mol,
    mol_b: Chem.rdchem.Mol,
    core: NDArray,
    forcefield: Forcefield,
    protein: Union[app.PDBFile, str],
    md_params: MDParams = DEFAULT_MD_PARAMS,
    n_windows: Optional[int] = None,
    min_overlap: Optional[float] = None,
    keep_idxs: Optional[List[int]] = None,
    min_cutoff: Optional[float] = 0.7,
):
    complex_sys, complex_conf, complex_box, complex_top, nwa = builders.build_protein_system(
        protein, forcefield.protein_ff, forcefield.water_ff
    )
    complex_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes, deboggle later
    complex_host_config = HostConfig(complex_sys, complex_conf, complex_box, nwa)
    complex_res = estimate_relative_free_energy_bisection(
        mol_a,
        mol_b,
        core,
        forcefield,
        complex_host_config,
        prefix="complex",
        md_params=md_params,
        n_windows=n_windows,
        min_overlap=min_overlap,
        keep_idxs=keep_idxs,
        min_cutoff=min_cutoff,
    )
    return complex_res, complex_top, complex_host_config


class Edge(NamedTuple):
    mol_a_name: str
    mol_b_name: str
    metadata: Dict[str, Any]


def get_failure_result_path(mol_a_name: str, mol_b_name: str):
    return f"failure_rbfe_result_{mol_a_name}_{mol_b_name}.pkl"


def get_success_result_path(mol_a_name: str, mol_b_name: str):
    return f"success_rbfe_result_{mol_a_name}_{mol_b_name}.pkl"


def run_edge_and_save_results(
    edge: Edge,
    mols: Dict[str, Chem.rdchem.Mol],
    forcefield: Forcefield,
    protein: app.PDBFile,
    file_client: AbstractFileClient,
    n_windows: Optional[int],
    md_params: MDParams = DEFAULT_MD_PARAMS,
):
    # Ensure that all mol props (e.g. _Name) are included in pickles
    # Without this get_mol_name(mol) will fail on roundtripped mol
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    try:
        mol_a = mols[edge.mol_a_name]
        mol_b = mols[edge.mol_b_name]

        all_cores = atom_mapping.get_cores(
            mol_a,
            mol_b,
            **DEFAULT_ATOM_MAPPING_KWARGS,
        )
        core = all_cores[0]

        complex_res, complex_top, _ = run_complex(
            mol_a,
            mol_b,
            core,
            forcefield,
            protein,
            md_params,
            n_windows=n_windows,
        )
        solvent_res, solvent_top, _ = run_solvent(
            mol_a,
            mol_b,
            core,
            forcefield,
            protein,
            md_params,
            n_windows=n_windows,
        )

    except Exception as err:
        print(
            "failed:",
            " | ".join(
                [
                    f"{edge.mol_a_name} -> {edge.mol_b_name} (kJ/mol)",
                    f"exp_ddg {edge.metadata['exp_ddg']:.2f}" if "exp_ddg" in edge.metadata else "",
                    f"fep_ddg {edge.metadata['fep_ddg']:.2f} +- {edge.metadata['fep_ddg_err']:.2f}"
                    if "fep_ddg" in edge.metadata and "fep_ddg_err" in edge.metadata
                    else "",
                ]
            ),
        )

        path = get_failure_result_path(edge.mol_a_name, edge.mol_b_name)
        tb = traceback.format_exception(None, err, err.__traceback__)
        file_client.store(path, pickle.dumps((edge, err, tb)))

        print(err)
        traceback.print_exc()

        return file_client.full_path(path)

    path = get_success_result_path(edge.mol_a_name, edge.mol_b_name)
    pkl_obj = (mol_a, mol_b, edge.metadata, core, solvent_res, solvent_top, complex_res, complex_top)
    file_client.store(path, pickle.dumps(pkl_obj))

    solvent_ddg = sum(solvent_res.final_result.dGs)
    solvent_ddg_err = np.linalg.norm(solvent_res.final_result.dG_errs)
    complex_ddg = sum(complex_res.final_result.dGs)
    complex_ddg_err = np.linalg.norm(complex_res.final_result.dG_errs)

    tm_ddg = complex_ddg - solvent_ddg
    tm_err = np.linalg.norm([complex_ddg_err, solvent_ddg_err])

    print(
        "finished:",
        " | ".join(
            [
                f"{edge.mol_a_name} -> {edge.mol_b_name} (kJ/mol)",
                f"complex {complex_ddg:.2f} +- {complex_ddg_err:.2f}",
                f"solvent {solvent_ddg:.2f} +- {solvent_ddg_err:.2f}",
                f"tm_pred {tm_ddg:.2f} +- {tm_err:.2f}",
                f"exp_ddg {edge.metadata['exp_ddg']:.2f}" if "exp_ddg" in edge.metadata else "",
                f"fep_ddg {edge.metadata['fep_ddg']:.2f} +- {edge.metadata['fep_ddg_err']:.2f}"
                if "fep_ddg" in edge.metadata and "fep_ddg_err" in edge.metadata
                else "",
            ]
        ),
    )

    return file_client.full_path(path)


def run_edges_parallel(
    ligands: Sequence[Chem.rdchem.Mol],
    edges: Sequence[Edge],
    ff: Forcefield,
    protein: app.PDBFile,
    n_gpus: int,
    pool_client: Optional[AbstractClient] = None,
    file_client: Optional[AbstractFileClient] = None,
    md_params: MDParams = DEFAULT_MD_PARAMS,
    n_windows: Optional[int] = None,
):
    mols = {get_mol_name(mol): mol for mol in ligands}

    pool_client = pool_client or CUDAPoolClient(n_gpus)
    pool_client.verify()

    file_client = file_client or FileClient()

    # Ensure that all mol props (e.g. _Name) are included in pickles
    # Without this get_mol_name(mol) will fail on roundtripped mol
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    jobs = [
        pool_client.submit(
            run_edge_and_save_results,
            edge,
            mols,
            ff,
            protein,
            file_client,
            n_windows,
            md_params,
        )
        for edge in edges
    ]

    # Remove references to completed jobs to allow garbage collection.
    # TODO: The current approach uses O(edges) memory in the worst case (e.g. if the first job gets stuck). Ideally we
    #   should process and remove references to jobs in the order they complete, but this would require an interface
    #   presently not implemented in our custom future classes.
    paths = []
    while jobs:
        job = jobs.pop(0)
        paths.append(job.result())

    return paths
