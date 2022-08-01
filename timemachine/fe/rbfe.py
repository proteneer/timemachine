import functools
import io
import warnings
from collections import namedtuple

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
from timemachine.md import minimizer
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

    bound_impls = [p.bound_impl(np.float32) for p in initial_state.U_fns]
    intg_impl = initial_state.integrator.impl()
    baro_impl = initial_state.barostat.impl(bound_impls)

    ctxt = custom_ops.Context(initial_state.x0, initial_state.v0, initial_state.box0, intg_impl, bound_impls, baro_impl)

    # burn-in
    ctxt.multiple_steps_U(
        lamb=initial_state.lamb,
        n_steps=protocol.burn_in,
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


class SimulationProtocol:
    def __init__(self, n_frames, burn_in=10000, steps_per_frame=1000):
        self.n_frames = n_frames
        self.burn_in = burn_in
        self.steps_per_frame = steps_per_frame


class InitialState:
    """
    An initial contains everything that is needed to bitwise reproduce a trajectory given a SimulationProtocol

    This object can be pickled safely.
    """

    def __init__(self, U_fns, integrator, barostat, x0, v0, box0, lamb):
        self.U_fns = U_fns
        self.integrator = integrator
        self.barostat = barostat
        self.x0 = x0
        self.v0 = v0
        self.box0 = box0
        self.lamb = lamb


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
        U_fns = hgs.get_U_fns()
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
        state = InitialState(U_fns, intg, baro, x0, v0, box0, lamb)
        initial_states.append(state)

    return initial_states


def generate_samples_for_all_states(initial_states, protocol):
    all_frames, all_boxes = [], []
    for state_idx, state in enumerate(initial_states):
        try:
            frames, boxes = sample(state, protocol)
        except Exception as exc:
            # propagate the line information
            raise ValueError(f"generate_samples_for_all_states failed on {state_idx}") from exc
        all_frames.append(frames)
        all_boxes.append(boxes)

    return all_frames, all_boxes


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


SimulationResult = namedtuple(
    "SimulationResult",
    [
        "all_dGs",
        "all_errs",
        "plot_png",
        "frames",
        "boxes",
        "initial_state",
        "simulation_protocol",
        "exception",
    ],
)


def estimate_free_energy_given_samples(all_frames, all_boxes, all_U_fns, temperature, lambda_schedule, prefix):
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
    all_frames: np.array of shape (L, T, N, 3)
        List of coordinates from all windows

    all_boxes: np.array of shape (L, T, 3, 3)
        List of boxes from all windows

    all_U_fns: list of energy functions of shape (L, P)
        List of energy functions from all windows

    temperature: float
        Temperature the system was run at

    lambda_schedule: np.array of shape (L,)
        Lambda schedule

    prefix: str
        A prefix to append to overlap figures

    Return
    ------
    3-tuple of dGs, dG_errs, overlap_plot
        All components have shape (L-1,)

    """
    # assume pair-BAR format
    # setup U_fns to operate in batch mode
    kT = BOLTZ * temperature
    beta = 1 / kT

    all_dGs = []
    all_errs = []

    all_batch_U_fns = []
    for lamb_idx, lamb in enumerate(lambda_schedule):
        bound_impls = [p.bound_impl(np.float32) for p in all_U_fns[lamb_idx]]
        batch_U_fns = get_batch_U_fns(bound_impls, lamb)
        all_batch_U_fns.append(batch_U_fns)

    U_names = []
    for U_fn in all_U_fns[0]:
        # convert from '<timemachine.lib.potentials.Nonbonded object at 0x7f7880b900b8>' -> Nonbonded
        U_names.append(repr(U_fn).split(".")[-1].split()[0])

    num_rows = len(lambda_schedule) - 1
    num_cols = len(U_names) + 1

    figure, all_axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 3))
    for lamb_idx, lamb in enumerate(lambda_schedule):
        if lamb_idx > 0:

            prev_frames = all_frames[lamb_idx - 1]
            cur_frames = all_frames[lamb_idx]

            prev_boxes = all_boxes[lamb_idx - 1]
            cur_boxes = all_boxes[lamb_idx]

            prev_U_fns = all_batch_U_fns[lamb_idx - 1]
            cur_U_fns = all_batch_U_fns[lamb_idx]

            # loop over bond, angle, torsion, nonbonded terms etc.
            all_fwd_delta_us = []
            all_rev_delta_us = []

            for u_idx, (prev_U_fn, cur_U_fn) in enumerate(zip(prev_U_fns, cur_U_fns)):
                fwd_delta_u = beta * (cur_U_fn(prev_frames, prev_boxes) - prev_U_fn(prev_frames, prev_boxes))
                rev_delta_u = beta * (prev_U_fn(cur_frames, cur_boxes) - cur_U_fn(cur_frames, cur_boxes))

                df, df_err = pymbar.BAR(fwd_delta_u, rev_delta_u)

                plot_axis = all_axes[lamb_idx - 1][u_idx]
                print(plot_axis)

                plot_BAR(df, df_err, fwd_delta_u, rev_delta_u, U_names[u_idx], plot_axis)

                all_fwd_delta_us.append(fwd_delta_u)
                all_rev_delta_us.append(rev_delta_u)

            # sanity check - I don't think the dG calculation commutes with its components, so we have to re-estimate
            # the dG from the sum of the delta_us as opposed to simply summing the component dGs
            total_fwd_delta_us = np.sum(all_fwd_delta_us, axis=0)
            total_rev_delta_us = np.sum(all_rev_delta_us, axis=0)
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

            print(
                f"{prefix} BAR: lambda {lambda_schedule[lamb_idx-1]:.3f} -> {lambda_schedule[lamb_idx]:.3f} dG: {total_dG:.3f} +- {total_dG_err:.3f} kJ/mol"
            )

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_as_bytes = buffer.read()

    return all_dGs, all_errs, img_as_bytes


def estimate_relative_free_energy(
    mol_a, mol_b, core, ff, host_config, seed, n_frames=1000, prefix="", lambda_schedule=None
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
    protocol = SimulationProtocol(n_frames=n_frames, burn_in=10000, steps_per_frame=1000)

    try:
        all_frames, all_boxes = generate_samples_for_all_states(initial_states, protocol)
        all_U_fns = [x.U_fns for x in initial_states]

        all_dGs, all_errs, plot_buffer = estimate_free_energy_given_samples(
            all_frames, all_boxes, all_U_fns, temperature, lambda_schedule, prefix
        )

        keep_frames = []
        keep_boxes = []
        keep_idxs = [0, -1]
        for idx in keep_idxs:
            keep_frames.append(all_frames[idx])
            keep_boxes.append(all_boxes[idx])

        return SimulationResult(all_dGs, all_errs, plot_buffer, keep_frames, keep_boxes, initial_states, protocol, None)

    except Exception as e:

        import traceback

        exception_msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print(f"Exception: {exception_msg}")

        return SimulationResult(None, None, None, None, None, initial_states, protocol, e)
