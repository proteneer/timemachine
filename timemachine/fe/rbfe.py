import functools
import io
import pickle
import warnings
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pymbar
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from timemachine.constants import BOLTZ, DEFAULT_TEMP
from timemachine.fe.single_topology_v3 import SingleTopologyV3
from timemachine.fe.system import convert_bps_into_system
from timemachine.fe.utils import get_mol_name, get_romol_conf
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops
from timemachine.lib.potentials import CustomOpWrapper
from timemachine.md import builders, minimizer
from timemachine.md.barostat.utils import get_bond_list, get_group_indices


def plot_atom_mapping_grid(mol_a, mol_b, core_smarts, core, show_idxs=False):
    mol_a_2d = Chem.Mol(mol_a)
    mol_b_2d = Chem.Mol(mol_b)
    mol_q_2d = Chem.MolFromSmarts(core_smarts)

    AllChem.Compute2DCoords(mol_q_2d)

    q_to_a = [[int(x[0]), int(x[1])] for x in enumerate(core[:, 0])]
    q_to_b = [[int(x[0]), int(x[1])] for x in enumerate(core[:, 1])]

    AllChem.GenerateDepictionMatching2DStructure(mol_a_2d, mol_q_2d, atomMap=q_to_a)
    AllChem.GenerateDepictionMatching2DStructure(mol_b_2d, mol_q_2d, atomMap=q_to_b)

    atom_colors_a = {}
    atom_colors_b = {}
    atom_colors_q = {}
    for c_idx, ((a_idx, b_idx), rgb) in enumerate(zip(core, np.random.random((len(core), 3)))):
        atom_colors_a[int(a_idx)] = tuple(rgb.tolist())
        atom_colors_b[int(b_idx)] = tuple(rgb.tolist())
        atom_colors_q[int(c_idx)] = tuple(rgb.tolist())

    if show_idxs:
        for atom in mol_a_2d.GetAtoms():
            atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
        for atom in mol_b_2d.GetAtoms():
            atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
        for atom in mol_q_2d.GetAtoms():
            atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))

    return Draw.MolsToGridImage(
        [mol_q_2d, mol_a_2d, mol_b_2d],
        molsPerRow=3,
        highlightAtomLists=[list(range(mol_q_2d.GetNumAtoms())), core[:, 0].tolist(), core[:, 1].tolist()],
        highlightAtomColors=[atom_colors_q, atom_colors_a, atom_colors_b],
        subImgSize=(300, 300),
        legends=["core", get_mol_name(mol_a), get_mol_name(mol_b)],
        useSVG=True,
    )


def get_batch_U_fns(bps, lamb):
    # return a function that takes in coords, boxes, lambda
    all_U_fns = []
    for bp in bps:

        def batch_U_fn(xs, boxes, bp_impl):
            Us = []
            for x, box in zip(xs, boxes):
                # tbd optimize to "selective" later
                _, _, U = bp_impl.execute(x, box, lamb)
                Us.append(U)
            return np.array(Us)

        # extra functools.partial is needed to deal with closure jank
        all_U_fns.append(functools.partial(batch_U_fn, bp_impl=bp))

    return all_U_fns


class HostConfig:
    def __init__(self, omm_system, conf, box):
        self.omm_system = omm_system
        self.conf = conf
        self.box = box


def sample(initial_state, protocol):
    """
    Generate a trajectory given an initial state and a simulation protocol
    """

    bound_impls = [p.bound_impl(np.float32) for p in initial_state.potentials]
    intg_impl = initial_state.integrator.impl()
    baro_impl = initial_state.barostat.impl(bound_impls)

    ctxt = custom_ops.Context(initial_state.x0, initial_state.v0, initial_state.box0, intg_impl, bound_impls, baro_impl)

    # burn-in
    ctxt.multiple_steps_U(
        lamb=initial_state.lamb,
        n_steps=protocol.n_eq_steps,
        lambda_windows=[initial_state.lamb],
        store_u_interval=0,
        store_x_interval=0,
    )

    # a crude, and probably not great, guess on the decorrelation time
    n_steps = protocol.n_frames * protocol.steps_per_frame
    all_nrgs, all_coords, all_boxes = ctxt.multiple_steps_U(
        lamb=initial_state.lamb,
        n_steps=n_steps,
        lambda_windows=[initial_state.lamb],
        store_u_interval=protocol.steps_per_frame,
        store_x_interval=protocol.steps_per_frame,
    )

    assert all_coords.shape[0] == protocol.n_frames
    assert all_boxes.shape[0] == protocol.n_frames

    return all_coords, all_boxes


@dataclass
class SimulationProtocol:
    n_frames: int
    n_eq_steps: int
    steps_per_frame: int


@dataclass
class InitialState:
    """
    An initial contains everything that is needed to bitwise reproduce a trajectory given a SimulationProtocol

    This object can be pickled safely.
    """

    potentials: List[CustomOpWrapper]
    integrator: LangevinIntegrator
    barostat: MonteCarloBarostat
    x0: np.ndarray
    v0: np.ndarray
    box0: np.ndarray
    lamb: float


@dataclass
class SimulationResult:
    all_dGs: List[np.ndarray]
    all_errs: List[float]
    overlaps_by_lambda: np.ndarray  # (L - 1,)
    overlaps_by_lambda_by_component: np.ndarray  # (len(U_names), L - 1)
    overlap_summary_png: bytes
    overlap_detail_png: bytes
    frames: List[np.ndarray]
    boxes: List[np.ndarray]
    initial_states: List[InitialState]
    protocol: SimulationProtocol


# setup the initial state so we can (hopefully) bitwise recover the identical simulation
# to help us debug errors.
def setup_initial_states(st, host_config, temperature, lambda_schedule, seed):

    host_bps, host_masses = openmm_deserializer.deserialize_system(host_config.omm_system, cutoff=1.2)
    host_conf = minimizer.minimize_host_4d(
        [st.mol_a, st.mol_b],
        host_config.omm_system,
        host_config.conf,
        st.ff,
        host_config.box,
    )

    initial_states = []

    for lamb_idx, lamb in enumerate(lambda_schedule):
        hgs = st.combine_with_host(convert_bps_into_system(host_bps), lamb=lamb)
        # minimize water box around the ligand by 4D-decoupling
        potentials = hgs.get_U_fns()
        mol_a_conf = get_romol_conf(st.mol_a)
        mol_b_conf = get_romol_conf(st.mol_b)
        ligand_conf = st.combine_confs(mol_a_conf, mol_b_conf)
        combined_conf = np.concatenate([host_conf, ligand_conf])
        x0 = combined_conf
        v0 = np.zeros_like(x0)
        box0 = host_config.box
        group_idxs = get_group_indices(get_bond_list(hgs.bond))
        run_seed = seed + lamb_idx
        combined_masses = np.concatenate([host_masses, st.combine_masses()])
        dt = 1e-3
        friction = 1.0
        intg = LangevinIntegrator(temperature, dt, friction, combined_masses, run_seed)
        baro = MonteCarloBarostat(len(combined_masses), 1.0, temperature, group_idxs, 15, run_seed + 1)
        state = InitialState(potentials, intg, baro, x0, v0, box0, lamb)
        initial_states.append(state)

    return initial_states


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

    plot_idx: triple (n_row, n_col, n_pos)
        where to place the subplot

    axes: matplotlib axis
        obj used to draw the figures

    """
    axes.set_title(f"{title}, dg: {df:.2f} +- {df_err:.2f} kTs")
    axes.hist(fwd_delta_u, alpha=0.5, label="fwd", density=True, bins=20)
    axes.hist(-rev_delta_u, alpha=0.5, label="-rev", density=True, bins=20)
    axes.set_xlabel("work (kTs)")
    axes.legend()


def plot_overlap_summary(ax, components, lambdas, overlaps):
    for component, ys in zip(components, overlaps):
        ax.plot(lambdas[:-1], ys, marker=".", label=component)

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
        Which states we keep samples for.

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

    stored_frames = []
    stored_boxes = []

    # memory complexity should be no more than that of 2-states worth of frames when generating samples needed to estimate the free energy.
    # appending too many idxs to keep_idxs may blow this up, so best to keep to first and last states in keep_idxs. when we change to multi-state
    # approaches later on this may need to change.
    prev_frames, prev_boxes = None, None
    prev_batch_U_fns = None

    # u_kln matrix (2, 2, n_frames) for each pair of adjacent lambda windows and energy term
    ukln_by_component_by_lambda = []

    for lamb_idx, initial_state in enumerate(initial_states):

        cur_frames, cur_boxes = sample(initial_state, protocol)
        bound_impls = [p.bound_impl(np.float32) for p in initial_state.potentials]
        cur_batch_U_fns = get_batch_U_fns(bound_impls, initial_state.lamb)

        if lamb_idx in keep_idxs:
            stored_frames.append(cur_frames)
            stored_boxes.append(cur_boxes)

        if lamb_idx > 0:

            ukln_by_component = []

            # loop over bond, angle, torsion, nonbonded terms etc.
            for u_idx, (prev_U_fn, cur_U_fn) in enumerate(zip(prev_batch_U_fns, cur_batch_U_fns)):
                x_cur = (cur_frames, cur_boxes)
                x_prev = (prev_frames, prev_boxes)
                u_00 = beta * prev_U_fn(*x_prev)
                u_01 = beta * prev_U_fn(*x_cur)
                u_10 = beta * cur_U_fn(*x_prev)
                u_11 = beta * cur_U_fn(*x_cur)
                ukln_by_component.append([[u_00, u_01], [u_10, u_11]])

                fwd_delta_u = u_10 - u_00
                rev_delta_u = u_01 - u_11
                df, df_err = pymbar.BAR(fwd_delta_u, rev_delta_u)
                plot_axis = all_axes[lamb_idx - 1][u_idx]
                plot_BAR(df, df_err, fwd_delta_u, rev_delta_u, U_names[u_idx], plot_axis)

            # sanity check - I don't think the dG calculation commutes with its components, so we have to re-estimate
            # the dG from the sum of the delta_us as opposed to simply summing the component dGs

            # (energy components, energy fxns = 2, sampled states = 2, frames)
            ukln_by_component = np.array(ukln_by_component, dtype=np.float64)
            total_fwd_delta_us = (ukln_by_component[:, 1, 0, :] - ukln_by_component[:, 0, 0, :]).sum(axis=0)
            total_rev_delta_us = (ukln_by_component[:, 0, 1, :] - ukln_by_component[:, 1, 1, :]).sum(axis=0)
            total_df, total_df_err = pymbar.BAR(total_fwd_delta_us, total_rev_delta_us)

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

    def pair_overlap(u_kln):
        assert u_kln.shape[:2] == (2, 2)
        u_kn = np.concatenate(u_kln, axis=1)
        N_k = u_kln.shape[2] * np.ones(u_kn.shape[0])
        return pymbar.MBAR(u_kn, N_k).computeOverlap()["matrix"][0, 1]  # type: ignore

    lambdas = [s.lamb for s in initial_states]

    _, (ax_top, ax_btm) = plt.subplots(2, 1, figsize=(7, 9))
    overlaps_by_lambda = np.array([pair_overlap(u_kln) for u_kln in ukln_by_lambda_by_component.sum(axis=0)])
    plot_overlap_summary(ax_top, ["Overall"], lambdas, [overlaps_by_lambda])

    overlaps_by_lambda_by_component = np.array(
        [[pair_overlap(u_kln) for u_kln in ukln_by_lambda] for ukln_by_lambda in ukln_by_lambda_by_component]
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
    keep_idxs=None,
    n_eq_steps=10000,
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

    host_config: HostConfig
        Configuration for the host system.

    n_frames: int
        number of samples to generate for each lambda windows, where each sample is 1000 steps of MD.

    prefix: str
        A prefix to append to figures

    seed: int
        Random seed to use for the simulations.

    lambda_schedule: list of float
        This should only be set when debugging or unit testing. This argument may be removed later.

    keep_idxs: list of int or None
        If None, return only the end-state frames. Otherwise if not None, use only for debugging, and this
        will return the frames corresponding to the idxs of interest.

    n_eq_steps: int
        Number of equilibration steps for each window.

    Returns
    -------
    SimulationResult
        Collected data from the simulation (see class for storage information). We currently return frames
        from only the first and last window.

    """
    single_topology = SingleTopologyV3(mol_a, mol_b, core, ff)

    if lambda_schedule is None:
        lambda_schedule = np.array([0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.11, 0.15, 0.20, 0.32, 0.42])
        lambda_schedule = np.concatenate([lambda_schedule, (1 - lambda_schedule[::-1])])
    else:
        warnings.warn("Warning: setting lambda_schedule manually, this argument may be removed in a future release.")

    temperature = DEFAULT_TEMP
    initial_states = setup_initial_states(single_topology, host_config, temperature, lambda_schedule, seed)
    protocol = SimulationProtocol(n_frames=n_frames, n_eq_steps=n_eq_steps, steps_per_frame=1000)

    if keep_idxs is None:
        keep_idxs = [0, -1]  # keep first and last frames
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


def run_pair(mol_a, mol_b, core, forcefield, protein, n_frames, seed, n_eq_steps=10000):
    box_width = 4.0
    solvent_sys, solvent_conf, solvent_box, solvent_top = builders.build_water_system(box_width)
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
    )

    complex_sys, complex_conf, _, _, complex_box, complex_top = builders.build_protein_system(protein)
    complex_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes, deboggle later
    complex_host_config = HostConfig(complex_sys, complex_conf, complex_box)
    complex_res = estimate_relative_free_energy(
        mol_a,
        mol_b,
        core,
        forcefield,
        complex_host_config,
        seed + 1,
        n_frames=n_frames,
        prefix="complex",
        n_eq_steps=n_eq_steps,
    )

    return solvent_res, solvent_top, complex_res, complex_top
