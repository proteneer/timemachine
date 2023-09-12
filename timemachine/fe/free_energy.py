from dataclasses import dataclass, replace
from functools import cache
from typing import Callable, List, Optional, Sequence, Tuple, Union, overload
from warnings import warn

import jax
import numpy as np
from numpy.typing import NDArray

from timemachine.constants import BOLTZ
from timemachine.fe import model_utils, topology
from timemachine.fe.bar import (
    bar_with_bootstrapped_uncertainty,
    df_and_err_from_u_kln,
    pair_overlap_from_ukln,
    works_from_ukln,
)
from timemachine.fe.energy_decomposition import (
    Batch_u_fn,
    EnergyDecomposedState,
    compute_energy_decomposed_u_kln,
    get_batch_u_fns,
)
from timemachine.fe.plots import make_dG_errs_figure, make_overlap_detail_figure, make_overlap_summary_figure
from timemachine.fe.protocol_refinement import greedy_bisection_step
from timemachine.fe.stored_arrays import StoredArrays
from timemachine.fe.utils import get_mol_masses, get_romol_conf
from timemachine.ff import ForcefieldParams
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops
from timemachine.md.barostat.utils import compute_box_center, get_bond_list, get_group_indices
from timemachine.md.hrex import HREX, HREXDiagnostics, ReplicaIdx, StateIdx, get_swap_attempts_per_iter_heuristic
from timemachine.md.states import CoordsVelBox
from timemachine.potentials import BoundPotential, HarmonicBond, SummedPotential
from timemachine.utils import batches, pairwise_transform_and_combine


class HostConfig:
    def __init__(self, omm_system, conf, box, num_water_atoms):
        self.omm_system = omm_system
        self.conf = conf
        self.box = box
        self.num_water_atoms = num_water_atoms


@dataclass(frozen=True)
class MDParams:
    n_frames: int
    n_eq_steps: int
    steps_per_frame: int
    seed: int
    local_steps: int = 0
    k: float = 1_000.0  # kJ/mol/nm^4
    min_radius: float = 1.0  # nm
    max_radius: float = 3.0  # nm
    freeze_reference: bool = True

    def __post_init__(self):
        assert self.steps_per_frame > 0
        assert self.n_frames > 0
        assert self.n_eq_steps >= 0
        assert 0.1 <= self.min_radius <= self.max_radius
        assert 0 <= self.local_steps <= self.steps_per_frame
        assert 1.0 <= self.k <= 1.0e6


@dataclass
class InitialState:
    """
    An initial contains everything that is needed to bitwise reproduce a trajectory given MDParams

    This object can be pickled safely.
    """

    potentials: List[BoundPotential]
    integrator: LangevinIntegrator
    barostat: Optional[MonteCarloBarostat]
    x0: np.ndarray
    v0: np.ndarray
    box0: np.ndarray
    lamb: float
    ligand_idxs: np.ndarray


@dataclass
class BarResult:
    dG: float
    dG_err: float
    dG_err_by_component: NDArray  # (len(U_names),)
    overlap: float
    overlap_by_component: NDArray  # (len(U_names),)
    u_kln_by_component: NDArray  # (len(U_names), 2, 2, N)


@dataclass
class PairBarPlots:
    dG_errs_png: bytes
    overlap_summary_png: bytes
    overlap_detail_png: bytes


@dataclass
class PairBarResult:
    """Results of BAR analysis on L-1 adjacent pairs of states given a sequence of L states."""

    initial_states: List[InitialState]  # length L
    bar_results: List[BarResult]  # length L - 1

    def __post_init__(self):
        assert len(self.bar_results) == len(self.initial_states) - 1

    @property
    def dGs(self) -> List[float]:
        return [r.dG for r in self.bar_results]

    @property
    def dG_errs(self) -> List[float]:
        return [r.dG_err for r in self.bar_results]

    @property
    def dG_err_by_component_by_lambda(self) -> NDArray:
        return np.array([r.dG_err_by_component for r in self.bar_results])

    @property
    def overlaps(self) -> List[float]:
        return [r.overlap for r in self.bar_results]

    @property
    def overlap_by_component_by_lambda(self) -> NDArray:
        return np.array([r.overlap_by_component for r in self.bar_results])

    @property
    def u_kln_by_component_by_lambda(self) -> NDArray:
        return np.array([r.u_kln_by_component for r in self.bar_results])


@dataclass
class SimulationResult:
    final_result: PairBarResult
    plots: PairBarPlots
    frames: List[StoredArrays]  # (len(keep_idxs), n_frames, N, 3)
    boxes: List[NDArray]
    md_params: MDParams
    intermediate_results: List[PairBarResult]
    hrex_diagnostics: Optional[HREXDiagnostics] = None


def image_frames(initial_state: InitialState, frames: np.ndarray, boxes: np.ndarray) -> np.ndarray:
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

    hb_potential = next(p.potential for p in initial_state.potentials if isinstance(p.potential, HarmonicBond))
    group_indices = get_group_indices(get_bond_list(hb_potential), len(initial_state.integrator.masses))
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
            topology.parameterize_nonbonded(
                ff_params.q_params,
                ff_params.q_params_intra,
                ff_params.q_params_solv,
                ff_params.lj_params,
                ff_params.lj_params_intra,
                ff_params.lj_params_solv,
                lamb,
            ),
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

    def prepare_host_edge(self, ff_params: ForcefieldParams, host_config: HostConfig, lamb: float):
        """
        Prepares the host-guest system

        Parameters
        ----------
        ff_params: ForcefieldParams
            forcefield parameters

        host_config: HostConfig
            HostConfig containing openmm System object to be deserialized.

        lamb: float
            alchemical parameter controlling 4D decoupling

        Returns
        -------
        3-tuple
            unbound_potentials, system_params, combined_masses

        """
        ligand_masses = get_mol_masses(self.mol)

        host_bps, host_masses = openmm_deserializer.deserialize_system(host_config.omm_system, cutoff=1.2)
        hgt = topology.HostGuestTopology(host_bps, self.top, host_config.num_water_atoms)

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


def get_context(initial_state: InitialState) -> custom_ops.Context:
    bound_impls = [p.to_gpu(np.float32).bound_impl for p in initial_state.potentials]
    intg_impl = initial_state.integrator.impl()
    baro_impl = initial_state.barostat.impl(bound_impls) if initial_state.barostat else None

    return custom_ops.Context(
        initial_state.x0,
        initial_state.v0,
        initial_state.box0,
        intg_impl,
        bound_impls,
        baro_impl,
    )


@overload
def sample(initial_state: InitialState, md_params: MDParams) -> Tuple[NDArray, NDArray, NDArray]:
    ...


@overload
def sample(
    initial_state: InitialState, md_params: MDParams, max_buffer_frames: int
) -> Tuple[StoredArrays, NDArray, NDArray]:
    ...


def sample(
    initial_state: InitialState, md_params: MDParams, max_buffer_frames: Optional[int] = None
) -> Tuple[Union[StoredArrays, NDArray], NDArray, NDArray]:
    """Generate a trajectory given an initial state and a simulation protocol

    Parameters
    ----------
    initial_state: InitialState
        (contains potentials, integrator, optional barostat)
    md_params: MDParams
        (specifies x0, v0, box0, number of MD steps, thinning interval, etc...)

    Returns
    -------
    xs, boxes: NDArray with .shape[0] = md_params.n_frames

    final_velocities: NDArray with shape (N,)

    Notes
    -----
    * Assertion error if coords become NaN
    """

    ctxt = get_context(initial_state)

    # burn-in
    if md_params.n_eq_steps:
        ctxt.multiple_steps(
            n_steps=md_params.n_eq_steps,
            store_x_interval=0,
        )

    rng = np.random.default_rng(md_params.seed)

    if md_params.local_steps > 0:
        ctxt.setup_local_md(initial_state.integrator.temperature, md_params.freeze_reference)

    assert np.all(np.isfinite(ctxt.get_x_t())), "Equilibration resulted in a nan"

    def run_production_steps(n_steps: int) -> Tuple[NDArray, NDArray, NDArray]:
        coords, boxes = ctxt.multiple_steps(
            n_steps=n_steps,
            store_x_interval=md_params.steps_per_frame,
        )
        final_velocities = ctxt.get_v_t()

        return coords, boxes, final_velocities

    def run_production_local_steps(n_steps: int) -> Tuple[NDArray, NDArray, NDArray]:
        coords = []
        boxes = []
        for steps in batches(n_steps, md_params.steps_per_frame):
            if steps < md_params.steps_per_frame:
                warn(
                    f"Batch of sample has {steps} steps, less than batch size {md_params.steps_per_frame}. Setting to {md_params.steps_per_frame}"
                )
                steps = md_params.steps_per_frame
            global_steps = steps - md_params.local_steps
            local_steps = md_params.local_steps
            if global_steps > 0:
                ctxt.multiple_steps(n_steps=global_steps)
            x_t, box_t = ctxt.multiple_steps_local(
                local_steps,
                initial_state.ligand_idxs.astype(np.int32),
                k=md_params.k,
                radius=rng.uniform(md_params.min_radius, md_params.max_radius),
                seed=rng.integers(np.iinfo(np.int32).max),
            )
            coords.append(x_t)
            boxes.append(box_t)

        final_velocities = ctxt.get_v_t()

        return np.concatenate(coords), np.concatenate(boxes), final_velocities

    all_coords: Union[NDArray, StoredArrays]

    steps_func = run_production_steps
    if md_params.local_steps > 0:
        steps_func = run_production_local_steps

    if max_buffer_frames is not None and max_buffer_frames > 0:
        all_coords = StoredArrays()
        all_boxes_: List[NDArray] = []
        final_velocities: NDArray = None  # type: ignore # work around "possibly unbound" error
        for n_frames in batches(md_params.n_frames, max_buffer_frames):
            batch_coords, batch_boxes, final_velocities = steps_func(n_frames * md_params.steps_per_frame)
            all_coords.extend(batch_coords)
            all_boxes_.extend(batch_boxes)
        all_boxes = np.array(all_boxes_)
    else:
        all_coords, all_boxes, final_velocities = steps_func(md_params.n_frames * md_params.steps_per_frame)

    assert len(all_coords) == md_params.n_frames
    assert len(all_boxes) == md_params.n_frames

    assert np.all(np.isfinite(all_coords[-1])), "Production resulted in a nan"

    return all_coords, all_boxes, final_velocities


class IndeterminateEnergyWarning(UserWarning):
    pass


def estimate_free_energy_bar(u_kln_by_component: NDArray, temperature: float) -> BarResult:
    """
    Estimate free energy difference for a pair of states given pre-generated samples.

    Parameters
    ----------
    u_kln_by_component: array
        u_kln in pymbar format (k = l = 2) for each energy component

    temperature: float
        Temperature

    Return
    ------
    PairBarResult
        results from BAR computation

    """

    # 1. We represent energies that we aren't able to evaluate (e.g. because of a fixed-point overflow in GPU potential code) with NaNs, but
    # 2. pymbar.MBAR will fail with LinAlgError if there are NaNs in the input.
    #
    # To work around this, we replace any NaNs with np.inf prior to the MBAR calculation.
    #
    # This is reasonable because u(x) -> inf corresponds to probability(x) -> 0, so this in effect declares that these
    # pathological states have zero weight.
    if np.any(np.isnan(u_kln_by_component)):
        warn(
            "Encountered NaNs in u_kln matrix. Replacing each instance with inf prior to MBAR calculation",
            IndeterminateEnergyWarning,
        )
        u_kln_by_component = np.where(np.isnan(u_kln_by_component), np.inf, u_kln_by_component)

    u_kln = u_kln_by_component.sum(0)

    df, df_err = bar_with_bootstrapped_uncertainty(u_kln)  # reduced units

    kBT = BOLTZ * temperature
    dG, dG_err = df * kBT, df_err * kBT  # kJ/mol

    overlap = pair_overlap_from_ukln(u_kln)

    # Componentwise calculations

    w_fwd_by_component, w_rev_by_component = jax.vmap(works_from_ukln)(u_kln_by_component)
    dG_err_by_component = np.array([df_and_err_from_u_kln(u_kln)[1] * kBT for u_kln in u_kln_by_component])

    # When forward and reverse works are identically zero (usually because a given energy term does not depend on
    # lambda, e.g. host-host nonbonded interactions), BAR error is undefined; we return 0.0 by convention.
    dG_err_by_component = np.where(
        np.all(np.isclose(w_fwd_by_component, 0.0), axis=1) & np.all(np.isclose(w_rev_by_component, 0.0), axis=1),
        0.0,
        dG_err_by_component,
    )

    overlap_by_component = np.array([pair_overlap_from_ukln(u_kln) for u_kln in u_kln_by_component])

    return BarResult(dG, dG_err, dG_err_by_component, overlap, overlap_by_component, u_kln_by_component)


def make_pair_bar_plots(res: PairBarResult, temperature: float, prefix: str) -> PairBarPlots:
    U_names = [type(p.potential).__name__ for p in res.initial_states[0].potentials]
    lambdas = [s.lamb for s in res.initial_states]

    overlap_detail_png = make_overlap_detail_figure(
        U_names, res.dGs, res.dG_errs, res.u_kln_by_component_by_lambda, temperature, prefix
    )

    dG_errs_png = make_dG_errs_figure(U_names, lambdas, res.dG_errs, res.dG_err_by_component_by_lambda)

    overlap_summary_png = make_overlap_summary_figure(
        U_names, lambdas, res.overlaps, res.overlap_by_component_by_lambda
    )

    return PairBarPlots(dG_errs_png, overlap_summary_png, overlap_detail_png)


def run_sims_sequential(
    initial_states: Sequence[InitialState],
    md_params: MDParams,
    temperature: float,
    keep_idxs: List[int],
) -> Tuple[PairBarResult, List[StoredArrays], List[NDArray]]:
    """Sequentially run simulations at each state in initial_states,
    returning summaries that can be used for pair BAR, energy decomposition, and other diagnostics

    Returns
    -------
    pair_bar_result: PairBarResult
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
    keep_idxs = [range(len(initial_states))[i] for i in keep_idxs]  # ensure all indices > 0
    assert np.all(np.array(keep_idxs) >= 0)

    for lamb_idx, initial_state in enumerate(initial_states):

        # run simulation
        cur_frames, cur_boxes, _ = sample(initial_state, md_params, max_buffer_frames=100)
        print(f"completed simulation at lambda={initial_state.lamb}!")

        # keep samples from any requested states in memory
        if lamb_idx in keep_idxs:
            stored_frames.append(cur_frames)
            stored_boxes.append(cur_boxes)

        bound_impls = [p.to_gpu(np.float32).bound_impl for p in initial_state.potentials]
        cur_batch_U_fns = get_batch_u_fns(bound_impls, temperature)

        state = EnergyDecomposedState(cur_frames, cur_boxes, cur_batch_U_fns)

        # analysis that depends on current and previous state
        if prev_state:
            state_pair = [prev_state, state]
            u_kln_by_component = compute_energy_decomposed_u_kln(state_pair)
            u_kln_by_component_by_lambda.append(u_kln_by_component)

        prev_state = state

    bar_results = [
        estimate_free_energy_bar(u_kln_by_component, temperature) for u_kln_by_component in u_kln_by_component_by_lambda
    ]

    return PairBarResult(list(initial_states), bar_results), stored_frames, stored_boxes


def make_batch_u_fns(initial_state: InitialState, temperature: float) -> List[Batch_u_fn]:
    assert initial_state.barostat is None or initial_state.barostat.temperature == temperature
    bound_impls = [p.to_gpu(np.float32).bound_impl for p in initial_state.potentials]
    return get_batch_u_fns(bound_impls, temperature)


class MinOverlapWarning(UserWarning):
    pass


def run_sims_bisection(
    initial_lambdas: Sequence[float],
    make_initial_state: Callable[[float], InitialState],
    md_params: MDParams,
    n_bisections: int,
    temperature: float,
    min_overlap: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[List[PairBarResult], List[StoredArrays], List[NDArray]]:
    r"""Starting from a specified lambda schedule, successively bisect the lambda interval between the pair of states
    with the lowest BAR overlap and sample the new state with MD.

    Parameters
    ----------
    initial_lambdas: sequence of float, length >= 2, monotonically increasing
        Initial protocol; starting point for bisection.

    make_initial_state: callable
        Function returning an InitialState (i.e., starting point for MD) given lambda

    md_params: MDParams
        Parameters used to simulate new states

    n_bisections: int
        Number of bisection steps to perform

    temperature: float
        Temperature in K

    min_overlap: float or None, optional
        If not None, return early when the BAR overlap between all neighboring pairs of states exceeds this value

    verbose: bool, optional
        Whether to print diagnostic information

    Returns
    -------
    results: list of IntermediateResult
        For each iteration of bisection, object containing the current list of states and array of energy-decomposed
        u_kln matrices.

    frames: list of StoredArrays
        Frames from the final iteration of bisection. Shape (L, F, N, 3) where N is the number of atoms.

    boxes: list of NDArray
        Boxes from the final iteration of bisection. Shape (L, F, 3, 3).
    """

    assert len(initial_lambdas) >= 2
    assert np.all(np.diff(initial_lambdas) > 0), "initial lambda schedule must be monotonically increasing"

    get_initial_state = cache(make_initial_state)

    @cache
    def get_samples(lamb: float) -> Tuple[StoredArrays, NDArray]:
        initial_state = get_initial_state(lamb)
        frames, boxes, _ = sample(initial_state, md_params, max_buffer_frames=100)
        return frames, boxes

    # NOTE: we don't cache get_state to avoid holding BoundPotentials in memory since they
    # 1. can use significant GPU memory
    # 2. can be reconstructed relatively quickly
    def get_state(lamb: float) -> EnergyDecomposedState[StoredArrays]:
        initial_state = get_initial_state(lamb)
        frames, boxes = get_samples(lamb)
        batch_u_fns = make_batch_u_fns(initial_state, temperature)
        return EnergyDecomposedState(frames, boxes, batch_u_fns)

    @cache
    def get_bar_result(lamb1: float, lamb2: float) -> BarResult:
        u_kln_by_component = compute_energy_decomposed_u_kln([get_state(lamb1), get_state(lamb2)])
        return estimate_free_energy_bar(u_kln_by_component, temperature)

    def overlap_to_cost(overlap: float) -> float:
        """Use -log(overlap) as the cost function for bisection; i.e., bisect the pair of states with lowest overlap."""
        return -np.log(overlap) if overlap != 0.0 else float("inf")

    def cost_to_overlap(cost: float) -> float:
        return np.exp(-cost)

    def cost_fn(lamb1: float, lamb2: float) -> float:
        overlap = get_bar_result(lamb1, lamb2).overlap
        return overlap_to_cost(overlap)

    def midpoint(x1: float, x2: float) -> float:
        return (x1 + x2) / 2.0

    def compute_intermediate_result(lambdas: Sequence[float]) -> PairBarResult:
        refined_initial_states = [get_initial_state(lamb) for lamb in lambdas]
        bar_results = [get_bar_result(lamb1, lamb2) for lamb1, lamb2 in zip(lambdas, lambdas[1:])]
        return PairBarResult(refined_initial_states, bar_results)

    lambdas = list(initial_lambdas)
    result = compute_intermediate_result(lambdas)
    results = [result]

    for iteration in range(n_bisections):

        if min_overlap is not None and np.all(np.array(result.overlaps) > min_overlap):
            if verbose:
                print(f"All BAR overlaps exceed min_overlap={min_overlap}. Returning after {iteration} iterations.")
            break

        lambdas_new, info = greedy_bisection_step(lambdas, cost_fn, midpoint)

        if verbose:
            costs, left_idx, lamb_new = info
            lamb1 = lambdas[left_idx]
            lamb2 = lambdas[left_idx + 1]
            print(
                f"Current minimum BAR overlap {cost_to_overlap(max(costs)):.3g} "
                f"between states at λ={lamb1:.3g} and λ={lamb2:.3g}. "
                f"Sampling new state at λ={lamb_new:.3g}…"
            )

        lambdas = lambdas_new
        result = compute_intermediate_result(lambdas)
        results.append(result)
    else:
        if min_overlap is not None:
            warn(
                f"Reached n_bisections={n_bisections} iterations without achieving min_overlap={min_overlap}. "
                f"The minimum BAR overlap was {np.min(result.overlaps)}.",
                MinOverlapWarning,
            )

    frames = [get_state(lamb).frames for lamb in lambdas]
    boxes = [get_state(lamb).boxes for lamb in lambdas]

    return results, frames, boxes


def run_sims_hrex(
    initial_states: Sequence[InitialState],
    md_params: MDParams,
    n_frames_per_iter: int,
    temperature: float,
    n_swap_attempts_per_iter: Optional[int] = None,
    print_diagnostics_interval: Optional[int] = 10,
) -> Tuple[PairBarResult, List[StoredArrays], List[NDArray], HREXDiagnostics]:
    r"""Sample from a sequence of states using nearest-neighbor Hamiltonian Replica EXchange (HREX).

    See documentation for :py:func:`timemachine.md.hrex.run_hrex` for details of the algorithm and implementation.

    Parameters
    ----------
    initial_states: sequence of InitialState
        States to sample. Should be ordered such that adjacent states have significant overlap for good mixing
        performance

    md_params: MDParams
        Parameters used to simulate new states

    n_frames_per_iter: int
        Number of frames to sample using MD per iteration

    temperature: float
        Temperature in K

    n_swap_attempts_per_iter: int or None, optional
        Number of nearest-neighbor swaps to attempt per iteration. Defaults to len(initial_states) ** 4.

    print_diagnostics_interval: int or None, optional
        If not None, print diagnostics every N iterations

    Returns
    -------
    PairBarResult
        results of pair BAR free energy analysis

    List[StoredArrays]
        coord trajectories

    List[NDArray]
        box trajectories

    HREXDiagnostics
        HREX statistics (e.g. swap rates, replica-state distribution)
    """

    if n_swap_attempts_per_iter is None:
        n_swap_attempts_per_iter = get_swap_attempts_per_iter_heuristic(len(initial_states))

    # Setting the numpy global PRNG state is necessary to ensure determinism in timemachine.md.moves.MonteCarloMove
    # TODO: Avoid use of global PRNG state (see https://github.com/proteneer/timemachine/issues/980)
    warn(f"Setting numpy global random state using seed {md_params.seed}")
    np.random.seed(md_params.seed)

    lambdas = [s.lamb for s in initial_states]
    initial_replicas = [CoordsVelBox(s.x0, s.v0, s.box0) for s in initial_states]

    bps = initial_states[0].potentials  # TODO: assert initial states have compatible potentials?
    sp = SummedPotential([bp.potential for bp in bps], [bp.params for bp in bps])
    ubp = sp.to_gpu(np.float32)

    def compute_log_q_matrix(xvbs: List[CoordsVelBox], params: List[NDArray]):
        coords = np.array([xvb.coords for xvb in xvbs])
        boxes = np.array([xvb.box for xvb in xvbs])
        _, _, U = ubp.unbound_impl.execute_selective_batch(coords, np.array(params), boxes, False, False, True)
        log_q = -U / (BOLTZ * temperature)
        return log_q

    def get_log_q_fn(xvb: List[CoordsVelBox]):
        def get_flattened_params(initial_state: InitialState) -> NDArray:
            return np.concatenate([bp.params.flatten() for bp in initial_state.potentials])

        params = [get_flattened_params(initial_state) for initial_state in initial_states]
        log_q_kl = compute_log_q_matrix(xvb, params)

        def log_q_fn(replica_idx: ReplicaIdx, state_idx: StateIdx) -> float:
            return log_q_kl[replica_idx, state_idx]

        return log_q_fn

    state_idxs = [StateIdx(i) for i, _ in enumerate(initial_states)]
    neighbor_pairs = list(zip(state_idxs, state_idxs[1:]))

    if len(initial_states) == 2:
        # Add an identity move to the mixture to ensure aperiodicity
        neighbor_pairs = [(StateIdx(0), StateIdx(0))] + neighbor_pairs

    hrex = HREX.from_replicas(initial_replicas)

    def get_equilibrated_initial_state(initial_state: InitialState) -> InitialState:
        if md_params.n_eq_steps == 0:
            return initial_state

        ctxt = get_context(initial_state)

        xs, boxes = ctxt.multiple_steps(md_params.n_eq_steps, store_x_interval=0)
        assert len(xs) == len(boxes) == 1
        x0 = xs[0]
        box0 = boxes[0]

        equilibrated_initial_state = replace(initial_state, x0=x0, box0=box0)
        return equilibrated_initial_state

    initial_states = [get_equilibrated_initial_state(initial_state) for initial_state in initial_states]

    samples_by_state_by_iter: List[List[Tuple[StoredArrays, NDArray]]] = []
    replica_idx_by_state_by_iter: List[List[ReplicaIdx]] = []
    fraction_accepted_by_pair_by_iter: List[List[Tuple[int, int]]] = []

    for iteration, n_frames_iter in enumerate(batches(md_params.n_frames, n_frames_per_iter), 1):

        def sample_replica(xvb: CoordsVelBox, state_idx: StateIdx) -> Tuple[StoredArrays, NDArray, NDArray]:
            initial_state = replace(initial_states[state_idx], x0=xvb.coords, v0=xvb.velocities, box0=xvb.box)
            md_params_replica = replace(
                md_params, n_frames=n_frames_iter, n_eq_steps=0, seed=np.random.randint(np.iinfo(np.int32).max)
            )

            # TODO: `sample` creates a new context from scratch. We should optimize this to reuse an existing context
            # with BoundPotential#set_params
            frames, boxes, final_velocities = sample(initial_state, md_params_replica, max_buffer_frames=100)

            return frames, boxes, final_velocities

        def replica_from_samples(samples: Tuple[StoredArrays, NDArray, NDArray]) -> CoordsVelBox:
            frames, boxes, final_velocities = samples
            return CoordsVelBox(frames[-1], final_velocities, boxes[-1])

        hrex, samples_by_state_ = hrex.sample_replicas(sample_replica, replica_from_samples)
        log_q = get_log_q_fn(hrex.replicas)
        hrex, fraction_accepted_by_pair = hrex.attempt_neighbor_swaps(neighbor_pairs, log_q, n_swap_attempts_per_iter)

        if len(initial_states) == 2:
            fraction_accepted_by_pair = fraction_accepted_by_pair[1:]  # remove stats for identity move

        samples_by_state = [(coords, boxes) for coords, boxes, _ in samples_by_state_]  # drop velocities
        samples_by_state_by_iter.append(samples_by_state)
        replica_idx_by_state_by_iter.append(hrex.replica_idx_by_state)
        fraction_accepted_by_pair_by_iter.append(fraction_accepted_by_pair)

        if print_diagnostics_interval and iteration % print_diagnostics_interval == 0:

            def get_swap_acceptance_rates(fraction_accepted_by_pair):
                return [
                    n_accepted / n_proposed if n_proposed else np.nan
                    for n_accepted, n_proposed in fraction_accepted_by_pair
                ]

            instantaneous_swap_acceptance_rates = get_swap_acceptance_rates(fraction_accepted_by_pair)
            average_swap_acceptance_rates = get_swap_acceptance_rates(np.sum(fraction_accepted_by_pair_by_iter, axis=0))

            def format_rate(r):
                return f"{r * 100.0:5.1f}%"

            def format_rates(rs):
                return " | ".join(format_rate(r) for r in rs)

            print("Completed HREX iteration", iteration)
            print("Acceptance rates (inst.)   :", format_rates(instantaneous_swap_acceptance_rates))
            print("Acceptance rates (average) :", format_rates(average_swap_acceptance_rates))
            print("Final replica permutation  :", hrex.replica_idx_by_state)
            print()

    # Concatenate frames and boxes from all iterations
    frames_by_state: List[StoredArrays] = [StoredArrays() for _ in lambdas]
    boxes_by_state_: List[List[NDArray]] = [[] for _ in lambdas]
    for samples_by_state in samples_by_state_by_iter:
        for samples, frames, boxes in zip(samples_by_state, frames_by_state, boxes_by_state_):
            frames_i, boxes_i = samples
            frames.extend(frames_i)
            boxes.extend(boxes_i)

    boxes_by_state: List[NDArray] = [np.array(boxes) for boxes in boxes_by_state_]

    def make_energy_decomposed_state(
        results: Tuple[StoredArrays, NDArray, InitialState]
    ) -> EnergyDecomposedState[StoredArrays]:
        frames, boxes, initial_state = results
        return EnergyDecomposedState(
            frames,
            boxes,
            get_batch_u_fns([pot.to_gpu(np.float32).bound_impl for pot in initial_state.potentials], temperature),
        )

    results_by_state = zip(frames_by_state, boxes_by_state, initial_states)

    bar_results = pairwise_transform_and_combine(
        results_by_state,
        make_energy_decomposed_state,
        lambda s1, s2: estimate_free_energy_bar(compute_energy_decomposed_u_kln([s1, s2]), temperature),
    )

    diagnostics = HREXDiagnostics(replica_idx_by_state_by_iter, fraction_accepted_by_pair_by_iter)

    return PairBarResult(initial_states, list(bar_results)), frames_by_state, boxes_by_state, diagnostics
