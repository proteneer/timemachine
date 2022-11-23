import functools
import io
import pickle
import traceback
import warnings
from typing import Any, Dict, NamedTuple, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pymbar
from rdkit import Chem
from simtk.openmm import app

from timemachine.constants import BOLTZ, DEFAULT_TEMP
from timemachine.fe import atom_mapping, model_utils
from timemachine.fe.bar import bar_with_bootstrapped_uncertainty
from timemachine.fe.free_energy import HostConfig, InitialState, SimulationProtocol, SimulationResult
from timemachine.fe.single_topology import SingleTopology
from timemachine.fe.system import convert_omm_system
from timemachine.fe.utils import get_mol_name, get_romol_conf
from timemachine.ff import Forcefield
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops
from timemachine.md import builders, minimizer
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.parallel.client import AbstractClient, AbstractFileClient, CUDAPoolClient, FileClient
from timemachine.potentials import jax_utils


def get_batch_U_fns(bps):
    # return a function that takes in coords, boxes
    all_U_fns = []
    for bp in bps:

        def batch_U_fn(xs, boxes, bp_impl):
            Us = []
            for x, box in zip(xs, boxes):
                # tbd optimize to "selective" later
                _, U = bp_impl.execute(x, box)
                Us.append(U)
            return np.array(Us)

        # extra functools.partial is needed to deal with closure jank
        all_U_fns.append(functools.partial(batch_U_fn, bp_impl=bp))

    return all_U_fns


def sample(initial_state, protocol):
    """
    Generate a trajectory given an initial state and a simulation protocol
    """

    bound_impls = [p.bound_impl(np.float32) for p in initial_state.potentials]
    intg_impl = initial_state.integrator.impl()
    if initial_state.barostat:
        baro_impl = initial_state.barostat.impl(bound_impls)
    else:
        baro_impl = None

    ctxt = custom_ops.Context(
        initial_state.x0,
        initial_state.v0,
        initial_state.box0,
        intg_impl,
        bound_impls,
        baro_impl,
    )

    # burn-in
    ctxt.multiple_steps_U(
        n_steps=protocol.n_eq_steps,
        store_u_interval=0,
        store_x_interval=0,
    )

    assert np.all(np.isfinite(ctxt.get_x_t())), "Equilibration resulted in a nan"

    # a crude, and probably not great, guess on the decorrelation time
    n_steps = protocol.n_frames * protocol.steps_per_frame
    _, all_coords, all_boxes = ctxt.multiple_steps_U(
        n_steps=n_steps,
        store_u_interval=0,
        store_x_interval=protocol.steps_per_frame,
    )

    assert all_coords.shape[0] == protocol.n_frames
    assert all_boxes.shape[0] == protocol.n_frames

    assert np.all(np.isfinite(all_coords[-1])), "Production resulted in a nan"

    return all_coords, all_boxes


# setup the initial state so we can (hopefully) bitwise recover the identical simulation
# to help us debug errors.
def setup_initial_states(st, host_config, temperature, lambda_schedule, seed, min_cutoff=0.7):
    """
    Setup the initial states for a series of lambda values. It is assumed that the lambda schedule
    is a monotonically increasing sequence in the closed interval [0,1].

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

    host = None
    if host_config:
        host_system, host_masses = convert_omm_system(host_config.omm_system)
        host_conf = minimizer.minimize_host_4d(
            [st.mol_a, st.mol_b],
            host_config.omm_system,
            host_config.conf,
            st.ff,
            host_config.box,
        )
        host = (host_system, host_masses, host_conf)

    initial_states = []

    # check that the lambda schedule is monotonically increasing.
    assert np.all(np.diff(lambda_schedule) > 0)

    for lamb_idx, lamb in enumerate(lambda_schedule):

        mol_a_conf = get_romol_conf(st.mol_a)
        mol_b_conf = get_romol_conf(st.mol_b)

        assert lamb >= 0.0 and lamb <= 1.0

        if lamb < 0.5:
            ligand_conf = st.combine_confs_lhs(mol_a_conf, mol_b_conf)
        else:
            ligand_conf = st.combine_confs_rhs(mol_a_conf, mol_b_conf)

        run_seed = seed + lamb_idx

        if host is not None:
            # run in an environment
            host_system, host_masses, host_conf = host
            system = st.combine_with_host(host_system, lamb=lamb)
            combined_masses = np.concatenate([host_masses, st.combine_masses()])
            potentials = system.get_U_fns()
            hmr_masses = model_utils.apply_hmr(combined_masses, system.bond.get_idxs())
            group_idxs = get_group_indices(get_bond_list(system.bond))
            baro = MonteCarloBarostat(len(hmr_masses), 1.0, temperature, group_idxs, 15, run_seed + 1)
            x0 = np.concatenate([host_conf, ligand_conf])
            box0 = host_config.box
        else:
            # run a vacuum simulation
            system = st.setup_intermediate_state(lamb)
            combined_masses = np.array(st.combine_masses())
            potentials = system.get_U_fns()
            hmr_masses = model_utils.apply_hmr(combined_masses, system.bond.get_idxs())
            baro = None
            x0 = ligand_conf
            box0 = np.eye(3, dtype=np.float64) * 10  # make a large 10x10x10nm box

        # minimize water box around the ligand by 4D-decoupling

        # hmr masses should be identical throughout the lambda schedule
        # bond idxs should be the same at the two end-states, note that a possible corner
        # case with bond breaking may seem to be problematic:

        # 0 1 2    0 1 2
        # C-O-C -> C.H-C

        # but this isn't an issue, since hydrogens will only ever be terminal atoms
        # and core hydrogens that are mapped to heavy atoms will take the mass of the
        # heavy atom (thereby not triggering the mass repartitioning to begin with).

        # but its reasonable to be skeptical, so we also assert consistency through the lambda
        # schedule as an extra sanity check.

        v0 = np.zeros_like(x0)  # tbd resample from Maxwell-boltzman?
        num_ligand_atoms = len(ligand_conf)
        num_total_atoms = len(x0)
        ligand_idxs = np.arange(num_total_atoms - num_ligand_atoms, num_total_atoms)

        dt = 2.5e-3
        friction = 1.0
        intg = LangevinIntegrator(temperature, dt, friction, hmr_masses, run_seed)

        state = InitialState(potentials, intg, baro, x0, v0, box0, lamb, ligand_idxs)
        initial_states.append(state)

    all_coords = optimize_coordinates(initial_states, min_cutoff=min_cutoff)
    for state, coords in zip(initial_states, all_coords):
        state.x0 = coords

    return initial_states


def plot_work(w_forward, w_reverse, axes):
    """histograms of +forward and -reverse works"""
    axes.hist(+w_forward, alpha=0.5, label="fwd", density=True, bins=20)
    axes.hist(-w_reverse, alpha=0.5, label="-rev", density=True, bins=20)
    axes.set_xlabel("work (kT)")
    axes.legend()


def plot_BAR(df, df_err, fwd_delta_u, rev_delta_u, title, axes):
    """
    Generate a subplot showing overlap for a particular pair of delta_us.

    Parameters
    ----------
    df: float
        reduced free energy

    df_err: float
        reduced free energy error

    fwd_delta_u: array
        reduced works

    rev_delta_u: array
        reduced reverse works

    title: str
        title to use

    axes: matplotlib axis
        obj used to draw the figures

    """
    axes.set_title(f"{title}, dg: {df:.2f} +- {df_err:.2f} kTs")
    plot_work(fwd_delta_u, rev_delta_u, axes)


def pair_overlap_from_ukln(u_kln):
    k, l, n = u_kln.shape
    assert k == l == 2
    u_kn = u_kln.reshape(k, -1)
    assert u_kn.shape == (k, l * n)
    N_k = n * np.ones(l)
    return 2 * pymbar.MBAR(u_kn, N_k).computeOverlap()["matrix"][0, 1]  # type: ignore


def plot_overlap_summary(ax, components, lambdas, overlaps):
    # one line per energy component
    for component, ys in zip(components, overlaps):
        percentages = 100 * ys
        ax.plot(lambdas[:-1], percentages, marker=".", label=component)

    # min and max within axis limits
    ax.set_ylim(-5, 105)
    ax.hlines([0, 100], min(lambdas), max(lambdas), color="grey", linestyles="--")

    # express in %
    ticks = [0, 25, 50, 75, 100]
    labels = [f"{t}%" for t in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)

    # labels and legends
    ax.set_xlabel(r"$\lambda_i$")
    ax.set_ylabel(r"pair BAR overlap ($\lambda_i$, $\lambda_{i+1}$)")
    ax.legend()


def estimate_free_energy_given_initial_states(initial_states, protocol, temperature, prefix, keep_idxs):
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

    protocol: Protocol
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
    # assume pair-BAR format
    kT = BOLTZ * temperature
    beta = 1 / kT

    all_dGs = []
    all_errs = []

    U_names = [type(U_fn).__name__ for U_fn in initial_states[0].potentials]

    num_rows = len(initial_states) - 1
    num_cols = len(U_names) + 1

    figure, all_axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 3))
    if num_rows == 1:
        all_axes = [all_axes]

    stored_frames = []
    stored_boxes = []

    # memory complexity should be no more than that of 2-states worth of frames when generating samples needed to estimate the free energy.
    # appending too many idxs to keep_idxs may blow this up, so best to keep to first and last states in keep_idxs. when we change to multi-state
    # approaches later on this may need to change.
    prev_frames, prev_boxes = None, None
    prev_batch_U_fns = None

    # u_kln matrix (2, 2, n_frames) for each pair of adjacent lambda windows and energy term
    ukln_by_component_by_lambda = []

    keep_idxs = keep_idxs or []
    if keep_idxs:
        assert all(np.array(keep_idxs) >= 0)

    for lamb_idx, initial_state in enumerate(initial_states):
        # Clear any old references to avoid holding on to objects in memory we don't need.
        cur_frames = None
        cur_boxes = None
        bound_impls = None
        cur_batch_U_fns = None
        cur_frames, cur_boxes = sample(initial_state, protocol)
        bound_impls = [p.bound_impl(np.float32) for p in initial_state.potentials]
        cur_batch_U_fns = get_batch_U_fns(bound_impls)

        if lamb_idx in keep_idxs:
            stored_frames.append(cur_frames)
            stored_boxes.append(cur_boxes)

        if lamb_idx > 0:

            ukln_by_component = []

            # loop over bond, angle, torsion, nonbonded terms etc.
            for u_idx, (prev_U_fn, cur_U_fn) in enumerate(zip(prev_batch_U_fns, cur_batch_U_fns)):
                u_00 = beta * prev_U_fn(prev_frames, prev_boxes)
                u_01 = beta * prev_U_fn(cur_frames, cur_boxes)
                u_10 = beta * cur_U_fn(prev_frames, prev_boxes)
                u_11 = beta * cur_U_fn(cur_frames, cur_boxes)
                ukln_by_component.append([[u_00, u_01], [u_10, u_11]])

                fwd_delta_u = u_10 - u_00
                rev_delta_u = u_01 - u_11
                plot_axis = all_axes[lamb_idx - 1][u_idx]
                plot_work(fwd_delta_u, rev_delta_u, plot_axis)
                plot_axis.set_title(U_names[u_idx])

            # sanity check - I don't think the dG calculation commutes with its components, so we have to re-estimate
            # the dG from the sum of the delta_us as opposed to simply summing the component dGs

            # (energy components, energy fxns = 2, sampled states = 2, frames)
            ukln_by_component = np.array(ukln_by_component, dtype=np.float64)
            total_fwd_delta_us = (ukln_by_component[:, 1, 0, :] - ukln_by_component[:, 0, 0, :]).sum(axis=0)
            total_rev_delta_us = (ukln_by_component[:, 0, 1, :] - ukln_by_component[:, 1, 1, :]).sum(axis=0)
            total_df, total_df_err = bar_with_bootstrapped_uncertainty(total_fwd_delta_us, total_rev_delta_us)

            plot_axis = all_axes[lamb_idx - 1][u_idx + 1]

            plot_BAR(
                total_df,
                total_df_err,
                total_fwd_delta_us,
                total_rev_delta_us,
                f"{prefix}_{lamb_idx-1}_to_{lamb_idx}",
                plot_axis,
            )

            total_dG = total_df / beta
            total_dG_err = total_df_err / beta

            all_dGs.append(total_dG)
            all_errs.append(total_dG_err)
            ukln_by_component_by_lambda.append(ukln_by_component)

            print(
                f"{prefix} BAR: lambda {lamb_idx-1} -> {lamb_idx} dG: {total_dG:.3f} +- {total_dG_err:.3f} kJ/mol",
                flush=True,
            )

        prev_frames = cur_frames
        prev_boxes = cur_boxes
        prev_batch_U_fns = cur_batch_U_fns

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    overlap_detail_png = buffer.read()

    # (energy components, lambdas, energy fxns = 2, sampled states = 2, frames)
    ukln_by_lambda_by_component = np.array(ukln_by_component_by_lambda).swapaxes(0, 1)

    lambdas = [s.lamb for s in initial_states]

    _, (ax_top, ax_btm) = plt.subplots(2, 1, figsize=(7, 9))
    overlaps_by_lambda = np.array([pair_overlap_from_ukln(u_kln) for u_kln in ukln_by_lambda_by_component.sum(axis=0)])
    plot_overlap_summary(ax_top, ["Overall"], lambdas, [overlaps_by_lambda])

    overlaps_by_lambda_by_component = np.array(
        [[pair_overlap_from_ukln(u_kln) for u_kln in ukln_by_lambda] for ukln_by_lambda in ukln_by_lambda_by_component]
    )
    plot_overlap_summary(ax_btm, U_names, lambdas, overlaps_by_lambda_by_component)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    overlap_summary_png = buffer.read()

    return SimulationResult(
        all_dGs,
        all_errs,
        overlaps_by_lambda,
        overlaps_by_lambda_by_component,
        overlap_summary_png,
        overlap_detail_png,
        stored_frames,
        stored_boxes,
        initial_states,
        protocol,
    )


def _optimize_coords_along_states(initial_states):
    # use the end-state to define the optimization settings
    end_state = initial_states[0]
    ligand_coords = end_state.x0[end_state.ligand_idxs]
    r_i = np.expand_dims(ligand_coords, axis=1)
    r_j = np.expand_dims(end_state.x0, axis=0)
    d_ij = np.linalg.norm(jax_utils.delta_r(r_i, r_j, box=end_state.box0), axis=-1)
    cutoff = 0.5  # in nanometers
    free_idxs = np.where(np.any(d_ij < cutoff, axis=0))[0].tolist()
    x_opt = end_state.x0
    x_traj = []
    for idx, initial_state in enumerate(initial_states):
        print(initial_state.lamb)
        bound_impls = [p.bound_impl(np.float32) for p in initial_state.potentials]
        val_and_grad_fn = minimizer.get_val_and_grad_fn(bound_impls, initial_state.box0)
        assert np.all(np.isfinite(x_opt)), "Initial coordinates contain nan or inf"
        # assert that the energy decreases only at the end-state.z
        if idx == 0:
            check_nrg = True
        else:
            check_nrg = False
        x_opt = minimizer.local_minimize(x_opt, val_and_grad_fn, free_idxs, assert_energy_decreased=check_nrg)
        x_traj.append(x_opt)
        assert np.all(np.isfinite(x_opt)), "Minimization resulted in a nan"
        del bound_impls

    return x_traj


def optimize_coordinates(initial_states, min_cutoff=0.7):
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

    for state, coords in zip(initial_states, all_xs):
        # sanity check that no atom has moved more than `min_cutoff` nm away
        assert (
            np.amax(np.linalg.norm(state.x0 - coords, axis=1)) < min_cutoff
        ), f"λ = {state.lamb} has minimized atom > {min_cutoff*10} Å from initial state"

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
        Collected data from the simulation (see class for storage information). We currently return frames
        from only the first and last window.

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
    combined_prefix = get_mol_name(mol_a) + "_" + get_mol_name(mol_b) + "_" + prefix
    try:
        return estimate_free_energy_given_initial_states(
            initial_states, protocol, temperature, combined_prefix, keep_idxs
        )
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
        core, smarts = atom_mapping.get_core_with_alignment(mol_a, mol_b, threshold=2.0)

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

        return path

    path = f"success_rbfe_result_{edge.mol_a_name}_{edge.mol_b_name}.pkl"
    pkl_obj = (mol_a, mol_b, edge.metadata, smarts, core, solvent_res, solvent_top, complex_res, complex_top)
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

    return path


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
    # should process and remove references to jobs in the order they complete, but this would require an interface
    # presently not implemented in our custom future classes.
    paths = []
    while jobs:
        job = jobs.pop(0)
        paths.append(job.result())

    return paths
