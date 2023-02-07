import pickle
import traceback
import warnings
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

import numpy as np
from rdkit import Chem
from simtk.openmm import app

from timemachine.constants import BOLTZ, DEFAULT_PRESSURE, DEFAULT_TEMP
from timemachine.fe import atom_mapping, model_utils
from timemachine.fe.bar import bar_with_bootstrapped_uncertainty, df_err_from_ukln, pair_overlap_from_ukln
from timemachine.fe.energy_decomposition import EnergyDecomposedState, compute_energy_decomposed_u_kln, get_batch_u_fns
from timemachine.fe.free_energy import HostConfig, InitialState, SimulationProtocol, SimulationResult, sample
from timemachine.fe.plots import make_dG_errs_figure, make_overlap_detail_figure, make_overlap_summary_figure
from timemachine.fe.single_topology import SingleTopology
from timemachine.fe.system import convert_omm_system
from timemachine.fe.utils import get_mol_name, get_romol_conf
from timemachine.ff import Forcefield
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat
from timemachine.md import builders, minimizer
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.parallel.client import AbstractClient, AbstractFileClient, CUDAPoolClient, FileClient
from timemachine.potentials import jax_utils


def setup_host(st: SingleTopology, host_config: HostConfig):

    host_system, host_masses = convert_omm_system(host_config.omm_system)
    host_conf = minimizer.minimize_host_4d(
        [st.mol_a, st.mol_b],
        host_config.omm_system,
        host_config.conf,
        st.ff,
        host_config.box,
    )
    host = (host_system, host_masses, host_conf)
    return host


def setup_in_vacuum(st, ligand_conf, lamb):

    system = st.setup_intermediate_state(lamb)
    combined_masses = np.array(st.combine_masses())

    potentials = system.get_U_fns()
    hmr_masses = model_utils.apply_hmr(combined_masses, system.bond.get_idxs())
    baro = None

    x0 = ligand_conf
    box0 = np.eye(3, dtype=np.float64) * 10  # make a large 10x10x10nm box

    return x0, box0, hmr_masses, potentials, baro


def setup_in_env(st, host, host_config, ligand_conf, lamb, temperature, run_seed):

    host_system, host_masses, host_conf = host

    system = st.combine_with_host(host_system, lamb=lamb)
    combined_masses = np.concatenate([host_masses, st.combine_masses()])

    potentials = system.get_U_fns()
    hmr_masses = model_utils.apply_hmr(combined_masses, system.bond.get_idxs())

    group_idxs = get_group_indices(get_bond_list(system.bond))
    baro = MonteCarloBarostat(len(hmr_masses), DEFAULT_PRESSURE, temperature, group_idxs, 15, run_seed + 1)

    x0 = np.concatenate([host_conf, ligand_conf])
    box0 = host_config.box

    return x0, box0, hmr_masses, potentials, baro


def assert_all_states_have_same_masses(initial_states: List[InitialState]):
    # hmr masses should be identical throughout the lambda schedule
    # bond idxs should be the same at the two end-states, note that a possible corner
    # case with bond breaking may seem to be problematic:

    # 0 1 2    0 1 2
    # C-O-C -> C.H-C

    # but this isn't an issue, since hydrogens will only ever be terminal atoms
    # and core hydrogens that are mapped to heavy atoms will take the mass of the
    # heavy atom (thereby not triggering the mass repartitioning to begin with).

    # but it's reasonable to be skeptical, so we also assert consistency through the lambda
    # schedule as an extra sanity check.
    masses = np.array([s.integrator.masses for s in initial_states])
    deviation_among_windows = masses.std(0)
    np.testing.assert_array_almost_equal(deviation_among_windows, 0, err_msg="masses assumed constant w.r.t. lambda")


def setup_initial_states(
    st,
    host_config,
    temperature,
    lambda_schedule,
    seed,
    min_cutoff=0.7,
):
    """
    Set up the initial states for a series of lambda values,
    so we can (hopefully) bitwise recover the identical simulation to debug errors.

    Assumes lambda schedule is a monotonically increasing sequence in the closed interval [0,1].

    Parameters
    ----------
    st: SingleTopology
        A single topology object

    host_config: HostConfig or None
        Configurations of the host. If None, then a vacuum state will be setup.

    temperature: float
        Temperature to run the simulation at.

    lambda_schedule: list of float of length K
        Lambda schedule.

    seed: int
        Random number seed

    min_cutoff: float
        throw error if any atom moves more than this distance (nm) after minimization

    Returns
    -------
    list of InitialStates
        Returns an initial state for each value of lambda.
    """

    if host_config:
        host = setup_host(st, host_config)
    else:
        host = None

    initial_states = []

    # check that the lambda schedule is monotonically increasing.
    assert np.all(np.diff(lambda_schedule) > 0)

    conf_a = get_romol_conf(st.mol_a)
    conf_b = get_romol_conf(st.mol_b)

    for lamb_idx, lamb in enumerate(lambda_schedule):
        ligand_conf = st.combine_confs(conf_a, conf_b, lamb)

        # use a different seed to initialize every window,
        # but in a way that should be symmetric for
        # A -> B vs. B -> A edge definitions
        init_seed = int(seed + hash(ligand_conf.tobytes())) % 10000

        if host is None:
            x0, box0, hmr_masses, potentials, baro = setup_in_vacuum(st, ligand_conf, lamb)
        else:
            x0, box0, hmr_masses, potentials, baro = setup_in_env(
                st, host, host_config, ligand_conf, lamb, temperature, init_seed
            )

        # provide a different run_seed for every lambda window,
        # but in a way that should be symmetric for
        # A -> B vs. B -> A edge definitions
        run_seed = int(seed + hash(bytes().join([np.array(p.params).tobytes() for p in potentials]))) % 10000
        # the constant is arbitrary, but see
        # https://github.com/proteneer/timemachine/commit/e1f7328f01f427534d8744aab6027338e116ad09

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

        # pack into state
        state = InitialState(potentials, intg, baro, x0, v0, box0, lamb, ligand_idxs)
        initial_states.append(state)

    # minimize environment atoms within min_cutoff of the ligand
    # optimization introduces dependencies among states with lam < 0.5, and among states with lam >= 0.5
    optimized_x0s = optimize_coordinates(initial_states, min_cutoff=min_cutoff)

    # update initial states in-place
    for state, x0 in zip(initial_states, optimized_x0s):
        state.x0 = x0

    # perform any concluding sanity-checks
    assert_all_states_have_same_masses(initial_states)

    return initial_states


def run_sequential_sims_given_initial_states(
    initial_states,
    protocol,
    temperature,
    keep_idxs,
):
    """Sequentially run simulations at each state in initial_states,
    returning summaries that can be used for pair BAR, energy decomposition, and other diagnostics

    Returns
    -------
    decomposed_u_klns: [n_lams - 1, n_components, 2, 2, n_frames] array
    stored_frames: coord trajectories, one for each state in keep_idxs
    stored_boxes: box trajectories, one for each state in keep_idxs

    Notes
    -----
    * Memory complexity:
        Memory demand should be no more than that of 2 states worth of frames.

        Requesting too many states in keep_idxs may blow this up,
        so best to keep to first and last states in keep_idxs.

        This restriction may need to be relaxed in the future if:
        * We decide to use MBAR(states) rather than sum_i BAR(states[i], states[i+1])
        * We use online protocol optimization approaches that require more states to be kept on-hand
    """
    stored_frames = []
    stored_boxes = []

    tmp_states = [None] * len(initial_states)

    # u_kln matrix (2, 2, n_frames) for each pair of adjacent lambda windows and energy term
    u_kln_by_component_by_lambda = []

    keep_idxs = keep_idxs or []
    if keep_idxs:
        assert all(np.array(keep_idxs) >= 0)

    for lamb_idx, initial_state in enumerate(initial_states):

        # run simulation
        cur_frames, cur_boxes = sample(initial_state, protocol)
        print(f"completed simulation at lambda={initial_state.lamb}!")

        # keep samples from any requested states in memory
        if lamb_idx in keep_idxs:
            stored_frames.append(cur_frames)
            stored_boxes.append(cur_boxes)

        # construct EnergyDecomposedState for current lamb_idx,
        # but keep no more than 2 of these states in memory at once
        if lamb_idx >= 2:
            tmp_states[lamb_idx - 2] = None

        bound_impls = [p.bound_impl(np.float32) for p in initial_state.potentials]
        cur_batch_U_fns = get_batch_u_fns(bound_impls, temperature)

        tmp_states[lamb_idx] = EnergyDecomposedState(cur_frames, cur_boxes, cur_batch_U_fns)

        # analysis that depends on current and previous state
        if lamb_idx > 0:
            state_pair = [tmp_states[lamb_idx - 1], tmp_states[lamb_idx]]
            u_kln_by_component = compute_energy_decomposed_u_kln(state_pair)
            u_kln_by_component_by_lambda.append(u_kln_by_component)

    return np.array(u_kln_by_component_by_lambda), stored_frames, stored_boxes


def estimate_free_energy_given_initial_states(
    initial_states,
    protocol,
    temperature,
    prefix,
    keep_idxs,
):
    """
    Estimate free energies given pre-generated samples. This implements the pair-BAR method, where
    windows assumed to be ordered with good overlap, with the final free energy being a sum
    of the components. The constants below are:

    L: the number of lambda windows
    T: the number of samples
    N: the number of atoms
    P: the number of components in the energy function.

    Parameters
    ----------
    initial_states: list of InitialState
        Initial state objects

    protocol: SimulationProtocol
        Detailing specifics of each simulation

    temperature: float
        Temperature the system was run at

    prefix: str
        A prefix that we append to the BAR overlap figures

    keep_idxs: list of int
        Which states we keep samples for. Must be positive.

    Return
    ------
    SimulationResult
        object containing results of the simulation

    """

    # run n_lambdas simulations in sequence
    u_kln_by_component_by_lambda, stored_frames, stored_boxes = run_sequential_sims_given_initial_states(
        initial_states, protocol, temperature, keep_idxs
    )

    # "pair BAR" free energy analysis
    kBT = BOLTZ * temperature
    beta = 1 / kBT

    all_dGs = []
    all_errs = []
    for lamb_idx, u_kln_by_component in enumerate(u_kln_by_component_by_lambda):
        # pair BAR
        u_kln = u_kln_by_component.sum(0)

        w_fwd = u_kln[1, 0] - u_kln[0, 0]
        w_rev = u_kln[0, 1] - u_kln[1, 1]

        df, df_err = bar_with_bootstrapped_uncertainty(w_fwd, w_rev)  # reduced units
        dG, dG_err = df / beta, df_err / beta  # kJ/mol

        message = f"{prefix} BAR: lambda {lamb_idx - 1} -> {lamb_idx} dG: {dG:.3f} +- {dG_err:.3f} kJ/mol"
        print(message, flush=True)

        all_dGs.append(dG)
        all_errs.append(dG_err)

    # (energy components, lambdas, energy fxns = 2, sampled states = 2, frames)
    ukln_by_lambda_by_component = np.array(u_kln_by_component_by_lambda).swapaxes(0, 1)

    # compute more diagnostics
    overlaps_by_lambda = [pair_overlap_from_ukln(u_kln) for u_kln in ukln_by_lambda_by_component.sum(axis=0)]

    dG_errs_by_lambda_by_component = np.array(
        [[df_err_from_ukln(u_kln) / beta for u_kln in ukln_by_lambda] for ukln_by_lambda in ukln_by_lambda_by_component]
    )
    overlaps_by_lambda_by_component = np.array(
        [[pair_overlap_from_ukln(u_kln) for u_kln in ukln_by_lambda] for ukln_by_lambda in ukln_by_lambda_by_component]
    )

    # generate figures
    U_names = [type(U_fn).__name__ for U_fn in initial_states[0].potentials]
    lambdas = [s.lamb for s in initial_states]

    overlap_detail_png = make_overlap_detail_figure(
        U_names,
        all_dGs,
        all_errs,
        overlaps_by_lambda,
        overlaps_by_lambda_by_component,
        u_kln_by_component_by_lambda,
        temperature,
        prefix,
    )
    dG_errs_png = make_dG_errs_figure(U_names, lambdas, all_errs, dG_errs_by_lambda_by_component)
    overlap_summary_png = make_overlap_summary_figure(
        U_names, lambdas, overlaps_by_lambda, overlaps_by_lambda_by_component
    )

    return SimulationResult(
        all_dGs,
        all_errs,
        dG_errs_by_lambda_by_component,
        overlaps_by_lambda,
        overlaps_by_lambda_by_component,
        dG_errs_png,
        overlap_summary_png,
        overlap_detail_png,
        stored_frames,
        stored_boxes,
        initial_states,
        protocol,
    )


def _optimize_coords_along_states(initial_states: List[InitialState]) -> List[np.ndarray]:

    # use the end-state to define the optimization settings
    end_state = initial_states[0]

    # select particles within cutoff of ligand
    x_opt = end_state.x0
    x_lig = x_opt[end_state.ligand_idxs]
    box = end_state.box0

    free_idxs = jax_utils.idxs_within_cutoff(x_opt, x_lig, box, cutoff=0.5).tolist()

    x_traj = []
    for idx, initial_state in enumerate(initial_states):
        print(initial_state.lamb)
        bound_impls = [p.bound_impl(np.float32) for p in initial_state.potentials]
        val_and_grad_fn = minimizer.get_val_and_grad_fn(bound_impls, initial_state.box0)
        assert np.all(np.isfinite(x_opt)), "Initial coordinates contain nan or inf"

        # only assert that the energy decreases at the end-state
        check_nrg = idx == 0

        x_opt = minimizer.local_minimize(x_opt, val_and_grad_fn, free_idxs, assert_energy_decreased=check_nrg)

        x_traj.append(x_opt)
        assert np.all(np.isfinite(x_opt)), "Minimization resulted in a nan"
        del bound_impls

    return x_traj


def optimize_coordinates(initial_states, min_cutoff=0.7) -> List[np.ndarray]:
    """
    Optimize geometries of the initial states.

    Parameters
    ----------
    initial_states: list of InitialState

    min_cutoff: float
        throw error if any atom moves more than this distance (nm) after minimization

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
        assert (
            displacement_distances < min_cutoff
        ).all(), f"λ = {state.lamb} moved an atom > {min_cutoff*10} Å from initial state during minimization"

    return all_xs


def estimate_relative_free_energy(
    mol_a,
    mol_b,
    core,
    ff,
    host_config,
    seed,
    n_frames=1000,
    prefix="",
    lambda_schedule=None,
    n_windows=None,
    keep_idxs=None,
    n_eq_steps=10000,
    steps_per_frame=400,
    min_cutoff=0.7,
):
    """
    Estimate relative free energy between mol_a and mol_b. Molecules should be aligned to each
    other and within the host environment.

    Parameters
    ----------
    mol_a: Chem.Mol
        initial molecule

    mol_b: Chem.Mol
        target molecule

    core: list of 2-tuples
        atom_mapping of atoms in mol_a into atoms in mol_b

    ff: ff.Forcefield
        Forcefield to be used for the system

    host_config: HostConfig or None
        Configuration for the host system. If None, then the vacuum leg is run.

    n_frames: int
        number of samples to generate for each lambda windows, where each sample is 1000 steps of MD.

    prefix: str
        A prefix to append to figures

    seed: int
        Random seed to use for the simulations.

    lambda_schedule: list of float
        This should only be set when debugging or unit testing. This argument may be removed later.

    n_windows: None
        Number of windows used for interpolating the the lambda schedule with additional windows.

    keep_idxs: list of int or None
        If None, return only the end-state frames. Otherwise if not None, use only for debugging, and this
        will return the frames corresponding to the idxs of interest.

    n_eq_steps: int
        Number of equilibration steps for each window.

    steps_per_frame: int
        The number of steps to take before collecting a frame

    min_cutoff: float
        throw error if any atom moves more than this distance (nm) after minimization

    Returns
    -------
    SimulationResult
        Collected data from the simulation (see class for storage information). Returned frames and boxes
        are defined by keep_idxs.

    """
    single_topology = SingleTopology(mol_a, mol_b, core, ff)

    if lambda_schedule is None:
        lambda_schedule = np.linspace(0, 1, n_windows or 30)
    else:
        assert n_windows is None
        warnings.warn("Warning: setting lambda_schedule manually, this argument may be removed in a future release.")

    temperature = DEFAULT_TEMP
    initial_states = setup_initial_states(
        single_topology, host_config, temperature, lambda_schedule, seed, min_cutoff=min_cutoff
    )
    protocol = SimulationProtocol(n_frames=n_frames, n_eq_steps=n_eq_steps, steps_per_frame=steps_per_frame)

    if keep_idxs is None:
        keep_idxs = [0, len(initial_states) - 1]  # keep first and last frames
    assert len(keep_idxs) <= len(lambda_schedule)

    # TODO: rename prefix to postfix, or move to beginning of combined_prefix?
    combined_prefix = get_mol_name(mol_a) + "_" + get_mol_name(mol_b) + "_" + prefix
    try:
        sim_result = estimate_free_energy_given_initial_states(
            initial_states, protocol, temperature, combined_prefix, keep_idxs
        )
        return sim_result
    except Exception as err:
        with open(f"failed_rbfe_result_{combined_prefix}.pkl", "wb") as fh:
            pickle.dump((initial_states, protocol, err), fh)
        raise err


def run_vacuum(
    mol_a,
    mol_b,
    core,
    forcefield,
    _,
    n_frames,
    seed,
    n_eq_steps=10000,
    steps_per_frame=400,
    n_windows=None,
    min_cutoff=1.5,
):
    # min_cutoff defaults to 15 Å since there is no environment to prevent conformational changes in the ligand
    vacuum_host_config = None
    return estimate_relative_free_energy(
        mol_a,
        mol_b,
        core,
        forcefield,
        vacuum_host_config,
        seed,
        n_frames=n_frames,
        prefix="vacuum",
        n_eq_steps=n_eq_steps,
        n_windows=n_windows,
        steps_per_frame=steps_per_frame,
        min_cutoff=min_cutoff,
    )


def run_solvent(
    mol_a,
    mol_b,
    core,
    forcefield,
    _,
    n_frames,
    seed,
    n_eq_steps=10000,
    steps_per_frame=400,
    n_windows=None,
    min_cutoff=0.7,
):
    box_width = 4.0
    solvent_sys, solvent_conf, solvent_box, solvent_top = builders.build_water_system(box_width, forcefield.water_ff)
    solvent_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes, deboggle later
    solvent_host_config = HostConfig(solvent_sys, solvent_conf, solvent_box)
    solvent_res = estimate_relative_free_energy(
        mol_a,
        mol_b,
        core,
        forcefield,
        solvent_host_config,
        seed,
        n_frames=n_frames,
        prefix="solvent",
        n_eq_steps=n_eq_steps,
        n_windows=n_windows,
        steps_per_frame=steps_per_frame,
        min_cutoff=min_cutoff,
    )
    return solvent_res, solvent_top


def run_complex(
    mol_a,
    mol_b,
    core,
    forcefield,
    protein,
    n_frames,
    seed,
    n_eq_steps=10000,
    steps_per_frame=400,
    n_windows=None,
    min_cutoff=0.7,
):
    complex_sys, complex_conf, _, _, complex_box, complex_top = builders.build_protein_system(
        protein, forcefield.protein_ff, forcefield.water_ff
    )
    complex_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes, deboggle later
    complex_host_config = HostConfig(complex_sys, complex_conf, complex_box)
    complex_res = estimate_relative_free_energy(
        mol_a,
        mol_b,
        core,
        forcefield,
        complex_host_config,
        seed,
        n_frames=n_frames,
        prefix="complex",
        n_eq_steps=n_eq_steps,
        n_windows=n_windows,
        steps_per_frame=steps_per_frame,
        min_cutoff=min_cutoff,
    )
    return complex_res, complex_top


class Edge(NamedTuple):
    mol_a_name: str
    mol_b_name: str
    metadata: Dict[str, Any]


def run_edge_and_save_results(
    edge: Edge,
    mols: Dict[str, Chem.rdchem.Mol],
    forcefield: Forcefield,
    protein: app.PDBFile,
    n_frames: int,
    seed: int,
    file_client: AbstractFileClient,
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
            ring_cutoff=0.12,
            chain_cutoff=0.2,
            max_visits=1e7,
            connected_core=True,
            max_cores=1e6,
            enforce_core_core=True,
            complete_rings=True,
            enforce_chiral=True,
            min_threshold=0,
        )
        core = all_cores[0]

        complex_res, complex_top = run_complex(mol_a, mol_b, core, forcefield, protein, n_frames, seed)
        solvent_res, solvent_top = run_solvent(mol_a, mol_b, core, forcefield, protein, n_frames, seed)

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

        path = f"failure_rbfe_result_{edge.mol_a_name}_{edge.mol_b_name}.pkl"
        tb = traceback.format_exception(None, err, err.__traceback__)
        file_client.store(path, pickle.dumps((edge, err, tb)))

        print(err)
        traceback.print_exc()

        return file_client.full_path(path)

    path = f"success_rbfe_result_{edge.mol_a_name}_{edge.mol_b_name}.pkl"
    pkl_obj = (mol_a, mol_b, edge.metadata, core, solvent_res, solvent_top, complex_res, complex_top)
    file_client.store(path, pickle.dumps(pkl_obj))

    solvent_ddg = np.sum(solvent_res.all_dGs)
    solvent_ddg_err = np.linalg.norm(solvent_res.all_errs)
    complex_ddg = np.sum(complex_res.all_dGs)
    complex_ddg_err = np.linalg.norm(complex_res.all_errs)

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
    n_frames: int,
    n_gpus: int,
    seed: int,
    pool_client: Optional[AbstractClient] = None,
    file_client: Optional[AbstractFileClient] = None,
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
            n_frames,
            seed + edge_idx,
            file_client,
        )
        for edge_idx, edge in enumerate(edges)
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
