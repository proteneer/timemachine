from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union, overload

import numpy as np
from numpy.typing import NDArray

from timemachine.constants import BOLTZ
from timemachine.fe import model_utils, topology
from timemachine.fe.bar import bar_with_bootstrapped_uncertainty, df_err_from_ukln, pair_overlap_from_ukln
from timemachine.fe.energy_decomposition import EnergyDecomposedState, compute_energy_decomposed_u_kln, get_batch_u_fns
from timemachine.fe.plots import make_dG_errs_figure, make_overlap_detail_figure, make_overlap_summary_figure
from timemachine.fe.protocol_refinement import greedy_bisection_step
from timemachine.fe.stored_arrays import StoredArrays
from timemachine.fe.utils import get_mol_masses, get_romol_conf
from timemachine.ff import ForcefieldParams
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops
from timemachine.lib.potentials import CustomOpWrapper, HarmonicBond
from timemachine.md.barostat.utils import compute_box_center, get_bond_list, get_group_indices


class HostConfig:
    def __init__(self, omm_system, conf, box):
        self.omm_system = omm_system
        self.conf = conf
        self.box = box


@dataclass
class MDParams:
    n_frames: int
    n_eq_steps: int
    steps_per_frame: int


@dataclass
class InitialState:
    """
    An initial contains everything that is needed to bitwise reproduce a trajectory given MDParams

    This object can be pickled safely.
    """

    potentials: List[CustomOpWrapper]
    integrator: LangevinIntegrator
    barostat: Optional[MonteCarloBarostat]
    x0: np.ndarray
    v0: np.ndarray
    box0: np.ndarray
    lamb: float
    ligand_idxs: np.ndarray


@dataclass
class PairBarResult:
    all_dGs: List[float]  # L - 1
    all_errs: List[float]  # L - 1
    dG_errs_by_lambda_by_component: NDArray  # (len(U_names), L - 1)
    overlaps_by_lambda: List[float]  # L - 1
    overlaps_by_lambda_by_component: NDArray  # (len(U_names), L - 1)
    u_kln_by_lambda_by_component: NDArray


@dataclass
class PairBarPlots:
    dG_errs_png: bytes
    overlap_summary_png: bytes
    overlap_detail_png: bytes


Frames = TypeVar("Frames")


@dataclass
class SimulationResult(Generic[Frames]):
    initial_states: List[InitialState]
    result: PairBarResult
    plots: PairBarPlots
    frames: List[Frames]
    boxes: List[NDArray]
    md_params: MDParams
    intermediate_results: List[Tuple[List[InitialState], PairBarResult]]


def image_frames(initial_state: InitialState, frames: Sequence[np.ndarray], boxes: np.ndarray) -> np.ndarray:
    """Images a sequence of frames within the periodic box given an Initial state. Recenters the simulation around the
    centroid of the coordinates specified by initial_state.ligand_idxs prior to imaging.

    Calling this function on a sequence of frames will NOT produce identical energies/du_dp/du_dx. Should only be used
    for visualization convenience.

    Parameters
    ----------

    initial_state: InitialState
        State that the frames came from

    frames: sequence of np.ndarray of coordinates
        Coordinates to image, sequence of K arrays with shape (N, 3)

    boxes: list of boxes
        Boxes to image coordinates into, list of K arrays with shape (3, 3)

    Returns
    -------
        imaged_coordinates
    """
    assert np.array(boxes).shape[1:] == (3, 3), "Boxes are not 3x3"
    assert len(frames) == len(boxes), "Number of frames and boxes don't match"

    hb_potential = next(p for p in initial_state.potentials if isinstance(p, HarmonicBond))
    group_indices = get_group_indices(get_bond_list(hb_potential))
    imaged_frames = np.empty_like(frames)
    for i, (frame, box) in enumerate(zip(frames, boxes)):
        assert frame.ndim == 2 and frame.shape[-1] == 3, "frames must have shape (N, 3)"
        # Recenter the frame around the centroid of the ligand
        ligand_centroid = np.mean(frame[initial_state.ligand_idxs], axis=0)
        center = compute_box_center(box)
        offset = ligand_centroid + center
        centered_frames = frame - offset

        imaged_frames[i] = model_utils.image_frame(group_indices, centered_frames, box)
    return np.array(imaged_frames)


class BaseFreeEnergy:
    @staticmethod
    def _get_system_params_and_potentials(ff_params: ForcefieldParams, topology, lamb: float):
        params_potential_pairs = [
            topology.parameterize_harmonic_bond(ff_params.hb_params),
            topology.parameterize_harmonic_angle(ff_params.ha_params),
            topology.parameterize_periodic_torsion(ff_params.pt_params, ff_params.it_params),
            topology.parameterize_nonbonded(ff_params.q_params, ff_params.q_params_intra, ff_params.lj_params, lamb),
        ]

        params, potentials = zip(*params_potential_pairs)
        return params, potentials


# this class is serializable.
class AbsoluteFreeEnergy(BaseFreeEnergy):
    def __init__(self, mol, top):
        """
        Compute the absolute free energy of a molecule via 4D decoupling.

        Parameters
        ----------
        mol: rdkit mol
            Ligand to be decoupled

        top: Topology
            topology.Topology to use

        """
        self.mol = mol
        self.top = top

    def prepare_host_edge(self, ff_params: ForcefieldParams, host_system, lamb: float):
        """
        Prepares the host-guest system

        Parameters
        ----------
        ff_params: ForcefieldParams
            forcefield parameters

        host_system: openmm.System
            openmm System object to be deserialized.

        lamb: float
            alchemical parameter controlling 4D decoupling

        Returns
        -------
        3-tuple
            unbound_potentials, system_params, combined_masses

        """
        ligand_masses = get_mol_masses(self.mol)

        host_bps, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)
        hgt = topology.HostGuestTopology(host_bps, self.top)

        final_params, final_potentials = self._get_system_params_and_potentials(ff_params, hgt, lamb)
        combined_masses = self._combine(ligand_masses, host_masses)
        return final_potentials, final_params, combined_masses

    def prepare_vacuum_edge(self, ff_params: ForcefieldParams):
        """
        Prepares the vacuum system

        Parameters
        ----------
        ff_params: ForcefieldParams
            forcefield parameters

        Returns
        -------
        3-tuple
            unbound_potentials, system_params, combined_masses

        """
        ligand_masses = get_mol_masses(self.mol)
        final_params, final_potentials = self._get_system_params_and_potentials(ff_params, self.top, 0.0)
        return final_potentials, final_params, ligand_masses

    def prepare_combined_coords(self, host_coords=None):
        """
        Returns the combined coordinates.

        Parameters
        ----------
        host_coords: np.array
            Nx3 array of atomic coordinates
            If None, return just the ligand coordinates.

        Returns
        -------
            combined_coordinates
        """
        ligand_coords = get_romol_conf(self.mol)
        return self._combine(ligand_coords, host_coords)

    def _combine(self, ligand_values, host_values=None):
        """
        Combine the values along the 0th axis.
        The host values will be first, if given.
        Then ligand values.

        Parameters
        ----------
        ligand_values: np.array
        host_values: Optional[np.array]

        Returns
        -------
            combined_values
        """
        if host_values is None:
            return ligand_values
        return np.concatenate([host_values, ligand_values])


def batches(n: int, batch_size: int) -> Iterable[int]:
    assert n >= 0
    assert batch_size > 0
    quot, rem = divmod(n, batch_size)
    for _ in range(quot):
        yield batch_size
    if rem:
        yield rem


@overload
def sample(initial_state: InitialState, md_params: MDParams) -> Tuple[NDArray, NDArray]:
    ...


@overload
def sample(initial_state: InitialState, md_params: MDParams, max_buffer_frames: int) -> Tuple[StoredArrays, NDArray]:
    ...


def sample(initial_state: InitialState, md_params: MDParams, max_buffer_frames: Optional[int] = None):
    """Generate a trajectory given an initial state and a simulation protocol

    Parameters
    ----------
    initial_state: InitialState
        (contains potentials, integrator, optional barostat)
    md_params: MDParams
        (specifies x0, v0, box0, number of MD steps, thinning interval, etc...)

    Returns
    -------
    xs, boxes: np.arrays with .shape[0] = md_params.n_frames

    Notes
    -----
    * Assertion error if coords become NaN
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
        n_steps=md_params.n_eq_steps,
        store_u_interval=0,
        store_x_interval=0,
    )

    assert np.all(np.isfinite(ctxt.get_x_t())), "Equilibration resulted in a nan"

    def run_production_steps(n_steps: int) -> Tuple[NDArray, NDArray]:
        _, coords, boxes = ctxt.multiple_steps_U(
            n_steps=n_steps,
            store_u_interval=0,
            store_x_interval=md_params.steps_per_frame,
        )
        return coords, boxes

    all_coords: Union[NDArray, StoredArrays]

    if max_buffer_frames:
        all_coords = StoredArrays()
        all_boxes_: List[NDArray] = []
        for n_frames in batches(md_params.n_frames, max_buffer_frames):
            batch_coords, batch_boxes = run_production_steps(n_frames * md_params.steps_per_frame)
            all_coords.extend(batch_coords)
            all_boxes_.extend(batch_boxes)
        all_boxes = np.array(all_boxes_)
    else:
        all_coords, all_boxes = run_production_steps(md_params.n_frames * md_params.steps_per_frame)

    assert len(all_coords) == md_params.n_frames
    assert len(all_boxes) == md_params.n_frames

    assert np.all(np.isfinite(all_coords[-1])), "Production resulted in a nan"

    return all_coords, all_boxes


def estimate_free_energy_pair_bar(
    u_kln_by_component_by_lambda: NDArray, temperature: float, prefix: str
) -> PairBarResult:
    """
    Estimate free energies given pre-generated samples. This implements the pair-BAR method, where
    windows assumed to be ordered with good overlap, with the final free energy being a sum
    of the components. The constants below are:

    Parameters
    ----------
    u_kln_by_component_by_lambda: array
        For each energy component and lambda pair, u_kln in pymbar format, where k = l = 2

    temperature: float
        Temperature the system was run at

    prefix: str
        A prefix that we append to the BAR overlap figures

    Return
    ------
    SimulationResult
        object containing results of the simulation

    """

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

        message = f"{prefix} BAR: lambda {lamb_idx} -> {lamb_idx + 1} dG: {dG:.3f} +- {dG_err:.3f} kJ/mol"
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

    return PairBarResult(
        all_dGs,
        all_errs,
        dG_errs_by_lambda_by_component,
        overlaps_by_lambda,
        overlaps_by_lambda_by_component,
        ukln_by_lambda_by_component,
    )


def make_pair_bar_plots(
    initial_states: List[InitialState], result: PairBarResult, temperature: float, prefix: str
) -> PairBarPlots:
    U_names = [type(U_fn).__name__ for U_fn in initial_states[0].potentials]
    lambdas = [s.lamb for s in initial_states]

    overlap_detail_png = make_overlap_detail_figure(
        U_names,
        result.all_dGs,
        result.all_errs,
        # u_kln_by_lambda_by_component -> u_kln_by_component_by_lambda
        result.u_kln_by_lambda_by_component.swapaxes(0, 1),
        temperature,
        prefix,
    )

    dG_errs_png = make_dG_errs_figure(U_names, lambdas, result.all_errs, result.dG_errs_by_lambda_by_component)

    overlap_summary_png = make_overlap_summary_figure(
        U_names, lambdas, result.overlaps_by_lambda, result.overlaps_by_lambda_by_component
    )

    return PairBarPlots(dG_errs_png, overlap_summary_png, overlap_detail_png)


def run_sims_sequential(
    initial_states: Sequence[InitialState],
    md_params: MDParams,
    temperature: float,
    keep_idxs: List[int],
) -> Tuple[NDArray, List[NDArray], List[NDArray]]:
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

    # keep no more than 2 states in memory at once
    prev_state: Optional[EnergyDecomposedState] = None

    # u_kln matrix (2, 2, n_frames) for each pair of adjacent lambda windows and energy term
    u_kln_by_component_by_lambda = []

    keep_idxs = keep_idxs or []
    if keep_idxs:
        assert all(np.array(keep_idxs) >= 0)

    for lamb_idx, initial_state in enumerate(initial_states):

        # run simulation
        cur_frames, cur_boxes = sample(initial_state, md_params)
        print(f"completed simulation at lambda={initial_state.lamb}!")

        # keep samples from any requested states in memory
        if lamb_idx in keep_idxs:
            stored_frames.append(cur_frames)
            stored_boxes.append(cur_boxes)

        bound_impls = [p.bound_impl(np.float32) for p in initial_state.potentials]
        cur_batch_U_fns = get_batch_u_fns(bound_impls, temperature)

        state = EnergyDecomposedState(cur_frames, cur_boxes, cur_batch_U_fns)

        # analysis that depends on current and previous state
        if prev_state:
            state_pair = [prev_state, state]
            u_kln_by_component = compute_energy_decomposed_u_kln(state_pair)
            u_kln_by_component_by_lambda.append(u_kln_by_component)

        prev_state = state

    return np.array(u_kln_by_component_by_lambda), stored_frames, stored_boxes


def sample_state(
    initial_state: InitialState, md_params: MDParams, temperature: float
) -> EnergyDecomposedState[StoredArrays]:
    frames, boxes = sample(initial_state, md_params, max_buffer_frames=1000)
    bound_impls = [p.bound_impl(np.float32) for p in initial_state.potentials]
    assert initial_state.barostat is None or initial_state.barostat.temperature == temperature
    batch_u_fns = get_batch_u_fns(bound_impls, temperature)
    return EnergyDecomposedState(frames, boxes, batch_u_fns)


def compute_bar_error(s1: EnergyDecomposedState, s2: EnergyDecomposedState) -> float:
    u_kln = compute_energy_decomposed_u_kln([s1, s2]).sum(axis=0)  # sum over components
    assert u_kln.shape[:2] == (2, 2)
    w_f = u_kln[1, 0] - u_kln[0, 0]
    w_r = u_kln[0, 1] - u_kln[1, 1]
    _, df_err = bar_with_bootstrapped_uncertainty(w_f, w_r)
    return df_err


def run_sims_with_greedy_bisection(
    initial_states: Sequence[InitialState],
    make_initial_state: Callable[[float], InitialState],
    md_params: MDParams,
    temperature: float,
) -> Tuple[List[Tuple[List[InitialState], NDArray]], List[StoredArrays], List[NDArray]]:

    assert len(initial_states) >= 2

    cache = lru_cache(maxsize=None)

    get_initial_state = cache(make_initial_state)

    @cache
    def get_state(lamb: float) -> EnergyDecomposedState[StoredArrays]:
        initial_state = get_initial_state(lamb)
        return sample_state(initial_state, md_params, temperature)

    @cache
    def bar_error(lamb1: float, lamb2: float) -> float:
        return compute_bar_error(get_state(lamb1), get_state(lamb2))

    def midpoint(x1, x2):
        return (x1 + x2) / 2.0

    lambdas = [initial_states[0].lamb, initial_states[-1].lamb]
    n_states_remaining = len(initial_states) - len(lambdas)
    results = []

    for _ in range(n_states_remaining):

        lambdas = greedy_bisection_step(lambdas, bar_error, midpoint)

        u_kln_by_component_by_lambda = np.array(
            [
                compute_energy_decomposed_u_kln([get_state(lamb1), get_state(lamb2)])
                for lamb1, lamb2 in zip(lambdas, lambdas[1:])
            ]
        )

        refined_initial_states = [get_initial_state(lamb) for lamb in lambdas]
        results.append((refined_initial_states, u_kln_by_component_by_lambda))

    frames = [get_state(lamb).frames for lamb in lambdas]
    boxes = [get_state(lamb).boxes for lamb in lambdas]

    return results, frames, boxes
