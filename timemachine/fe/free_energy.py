from dataclasses import dataclass, replace
from functools import cache
from typing import Callable, List, Optional, Sequence, Tuple, Union
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
from timemachine.fe.energy_decomposition import EnergyDecomposedState, compute_energy_decomposed_u_kln, get_batch_u_fns
from timemachine.fe.plots import (
    plot_as_png_fxn,
    plot_dG_errs_figure,
    plot_overlap_detail_figure,
    plot_overlap_summary_figure,
)
from timemachine.fe.protocol_refinement import greedy_bisection_step
from timemachine.fe.stored_arrays import StoredArrays
from timemachine.fe.utils import get_mol_masses, get_romol_conf
from timemachine.ff import ForcefieldParams
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops
from timemachine.lib.custom_ops import Context
from timemachine.md.barostat.utils import compute_box_center, get_bond_list, get_group_indices
from timemachine.md.exchange.exchange_mover import get_water_idxs
from timemachine.md.hrex import (
    HREX,
    HREXDiagnostics,
    HREXPlots,
    ReplicaIdx,
    StateIdx,
    get_swap_attempts_per_iter_heuristic,
)
from timemachine.md.states import CoordsVelBox
from timemachine.potentials import BoundPotential, HarmonicBond, NonbondedInteractionGroup, SummedPotential
from timemachine.utils import batches, pairwise_transform_and_combine

WATER_SAMPLER_MOVERS = (
    custom_ops.TIBDExchangeMove_f32,
    custom_ops.TIBDExchangeMove_f64,
)


class HostConfig:
    def __init__(self, omm_system, conf, box, num_water_atoms):
        self.omm_system = omm_system
        self.conf = conf
        self.box = box
        self.num_water_atoms = num_water_atoms


@dataclass(frozen=True)
class HREXParams:
    """
    Parameters
    ----------

    n_frames_bisection:
        Number of frames to sample using MD during the initial bisection phase used to determine lambda spacing

    n_frames_per_iter:
        Number of frames to sample using MD per HREX iteration.
    """

    n_frames_bisection: int = 100
    n_frames_per_iter: int = 1

    def __post_init__(self):
        assert self.n_frames_bisection > 0
        assert self.n_frames_per_iter > 0


@dataclass(frozen=True)
class WaterSamplingParams:
    """
    Parameters
    ----------

    interval:
        How many steps of MD between water sampling moves

    n_proposals:
        Number of proposals per make.

    batch_size:
        Internal parameter detailing the parallelism of the mover, typically can be left at default.

    radius:
        Radius, in nanometers, from the centroid of the molecule to treat as the inner target volume
    """

    interval: int = 400
    n_proposals: int = 1000
    batch_size: int = 250
    radius: float = 1.0

    def __post_init__(self):
        assert self.interval > 0
        assert self.n_proposals > 0
        assert self.radius > 0.0
        assert self.batch_size > 0
        assert self.batch_size <= self.n_proposals


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

    # Set to HREXParams or None to disable HREX
    hrex_params: Optional[HREXParams] = None
    # Setting water_sampling_params to None disables water sampling.
    water_sampling_params: Optional[WaterSamplingParams] = None

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
    x0: NDArray
    v0: NDArray
    box0: NDArray
    lamb: float
    ligand_idxs: NDArray
    protein_idxs: NDArray

    def __post_init__(self):
        assert self.ligand_idxs.dtype == np.int32 or self.ligand_idxs.dtype == np.int64
        assert self.protein_idxs.dtype == np.int32 or self.protein_idxs.dtype == np.int64


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
class Trajectory:
    frames: StoredArrays  # (frame, atom, dim)
    boxes: List[NDArray]  # (frame, dim, dim)
    final_velocities: Optional[NDArray]  # (atom, dim)
    final_barostat_volume_scale_factor: Optional[float] = None

    def __post_init__(self):
        n_frames = len(self.frames)
        assert len(self.boxes) == n_frames
        if n_frames == 0:
            return
        n_atoms, n_dims = self.frames[0].shape
        assert self.boxes[0].shape == (n_dims, n_dims)
        if self.final_velocities is not None:
            assert self.final_velocities.shape == (n_atoms, n_dims)

    def extend(self, other: "Trajectory"):
        """Concatenate another trajectory to the end of this one"""
        self.frames.extend(other.frames)
        self.boxes.extend(other.boxes)
        self.final_velocities = other.final_velocities
        self.final_barostat_volume_scale_factor = other.final_barostat_volume_scale_factor

    @classmethod
    def empty(cls):
        return Trajectory(StoredArrays(), [], None, None)


@dataclass
class SimulationResult:
    final_result: PairBarResult
    plots: PairBarPlots
    trajectories: List[Trajectory]
    md_params: MDParams
    intermediate_results: List[PairBarResult]
    hrex_diagnostics: Optional[HREXDiagnostics] = None
    hrex_plots: Optional[HREXPlots] = None

    @property
    def frames(self) -> List[StoredArrays]:
        return [traj.frames for traj in self.trajectories]

    @property
    def boxes(self) -> List[NDArray]:
        return [np.array(traj.boxes) for traj in self.trajectories]


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


def get_water_sampler_params(initial_state: InitialState) -> NDArray:
    """Given an initial state, return the parameters that define the nonbonded parameters of water with respect to the
    entire system.

    Since we split the different components of the NB into ligand-water, ligand-protein, we want to use the ligand-water parameter
    to ensure that the ligand component is correctly configured for the specific lambda window.
    """
    summed_pot = next(p.potential for p in initial_state.potentials if isinstance(p.potential, SummedPotential))
    # TBD Figure out a better way of handling the jankyness of having to select the IxnGroup that defines the Ligand-Water
    # Currently this just hardcodes to the first ixn group potential which is the ligand-water ixn group as returned by HostTopology.parameterize_nonbonded
    ixn_group_idx = next(i for i, pot in enumerate(summed_pot.potentials) if isinstance(pot, NonbondedInteractionGroup))
    assert isinstance(summed_pot.potentials[ixn_group_idx], NonbondedInteractionGroup)
    water_params = summed_pot.params_init[ixn_group_idx]
    assert water_params.shape[1] == 4
    water_params = np.asarray(water_params)
    return water_params


def get_context(initial_state: InitialState, md_params: Optional[MDParams] = None) -> Context:
    """
    Construct a Context that has a single SummedPotential that combines the potentials defined by the initial state
    """
    potentials = [bp.potential for bp in initial_state.potentials]
    params = [bp.params for bp in initial_state.potentials]
    potential = SummedPotential(potentials, params).to_gpu(np.float32)

    # Set up context for MD using overall potential
    bound_impl = potential.bind_params_list(params).bound_impl
    bound_impls = [bound_impl]
    intg_impl = initial_state.integrator.impl()
    movers = []
    if initial_state.barostat:
        movers.append(initial_state.barostat.impl(bound_impls))
    if md_params is not None and md_params.water_sampling_params is not None:
        # Setup the water indices
        hb_potential = next(p.potential for p in initial_state.potentials if isinstance(p.potential, HarmonicBond))
        group_indices = get_group_indices(get_bond_list(hb_potential), len(initial_state.integrator.masses))

        water_idxs = get_water_idxs(group_indices, ligand_idxs=initial_state.ligand_idxs)

        summed_pot = next(p.potential for p in initial_state.potentials if isinstance(p.potential, SummedPotential))
        # Select a Nonbonded Potential to get the the cutoff/beta, assumes all have same cutoff/beta.
        nb = next(p for p in summed_pot.potentials if isinstance(p, NonbondedInteractionGroup))

        water_params = get_water_sampler_params(initial_state)

        # Generate a new random seed based on the md params seed
        rng = np.random.default_rng(md_params.seed)
        water_sampler_seed = rng.integers(np.iinfo(np.int32).max)

        water_sampler = custom_ops.TIBDExchangeMove_f32(
            initial_state.x0.shape[0],
            initial_state.ligand_idxs.tolist(),
            [water_group.tolist() for water_group in water_idxs],
            water_params,
            initial_state.integrator.temperature,
            nb.beta,
            nb.cutoff,
            md_params.water_sampling_params.radius,
            water_sampler_seed,
            md_params.water_sampling_params.n_proposals,
            md_params.water_sampling_params.interval,
            batch_size=md_params.water_sampling_params.batch_size,
        )
        movers.append(water_sampler)

    return Context(initial_state.x0, initial_state.v0, initial_state.box0, intg_impl, bound_impls, movers=movers)


def sample_with_context(
    ctxt: Context, md_params: MDParams, temperature: float, ligand_idxs: NDArray, max_buffer_frames: int
) -> Trajectory:
    # burn-in
    if md_params.n_eq_steps:
        # Set barostat interval to 15 for equilibration, then back to the original interval for production
        barostat = ctxt.get_barostat()
        original_interval = 0 if barostat is None else barostat.get_interval()
        equil_barostat_interval = 15
        if barostat is not None:
            barostat.set_interval(equil_barostat_interval)
        ctxt.multiple_steps(
            n_steps=md_params.n_eq_steps,
            store_x_interval=0,
        )
        if barostat is not None:
            barostat.set_interval(original_interval)

    rng = np.random.default_rng(md_params.seed)

    if md_params.local_steps > 0:
        ctxt.setup_local_md(temperature, md_params.freeze_reference)

    assert np.all(np.isfinite(ctxt.get_x_t())), "Equilibration resulted in a nan"

    def run_production_steps(n_steps: int) -> Tuple[NDArray, NDArray, NDArray]:
        coords, boxes = ctxt.multiple_steps(
            n_steps=n_steps,
            store_x_interval=md_params.steps_per_frame,
        )
        assert np.all(coords[-1] == ctxt.get_x_t())
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
                ligand_idxs.astype(np.int32),
                k=md_params.k,
                radius=rng.uniform(md_params.min_radius, md_params.max_radius),
                seed=rng.integers(np.iinfo(np.int32).max),
            )
            coords.append(x_t)
            boxes.append(box_t)

        assert np.all(coords[-1][-1] == ctxt.get_x_t())
        final_velocities = ctxt.get_v_t()

        return np.concatenate(coords), np.concatenate(boxes), final_velocities

    all_coords: Union[NDArray, StoredArrays]

    steps_func = run_production_steps
    if md_params.local_steps > 0:
        steps_func = run_production_local_steps

    all_coords = StoredArrays()
    all_boxes: List[NDArray] = []
    final_velocities: NDArray = None  # type: ignore # work around "possibly unbound" error
    for n_frames in batches(md_params.n_frames, max_buffer_frames):
        batch_coords, batch_boxes, final_velocities = steps_func(n_frames * md_params.steps_per_frame)
        all_coords.extend(batch_coords)
        all_boxes.extend(batch_boxes)

    assert len(all_coords) == md_params.n_frames
    assert len(all_boxes) == md_params.n_frames

    assert np.all(np.isfinite(all_coords[-1])), "Production resulted in a nan"

    final_barostat_volume_scale_factor = ctxt.get_barostat().get_volume_scale_factor() if ctxt.get_barostat() else None

    return Trajectory(all_coords, all_boxes, final_velocities, final_barostat_volume_scale_factor)


def sample(initial_state: InitialState, md_params: MDParams, max_buffer_frames: int) -> Trajectory:
    """Generate a trajectory given an initial state and a simulation protocol

    Parameters
    ----------
    initial_state: InitialState
        (contains potentials, integrator, optional barostat)

    md_params: MDParams
        MD parameters

    max_buffer_frames: int
        number of frames to store in memory before dumping to disk

    Returns
    -------
    Trajectory

    Notes
    -----
    * Assertion error if coords become NaN
    """

    ctxt = get_context(initial_state, md_params)

    return sample_with_context(
        ctxt, md_params, initial_state.integrator.temperature, initial_state.ligand_idxs, max_buffer_frames
    )


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

    overlap_detail_png = plot_as_png_fxn(
        plot_overlap_detail_figure, U_names, res.dGs, res.dG_errs, res.u_kln_by_component_by_lambda, temperature, prefix
    )

    dG_errs_png = plot_as_png_fxn(plot_dG_errs_figure, U_names, lambdas, res.dG_errs, res.dG_err_by_component_by_lambda)

    overlap_summary_png = plot_as_png_fxn(
        plot_overlap_summary_figure, U_names, lambdas, res.overlaps, res.overlap_by_component_by_lambda
    )

    return PairBarPlots(dG_errs_png, overlap_summary_png, overlap_detail_png)


def run_sims_sequential(
    initial_states: Sequence[InitialState],
    md_params: MDParams,
    temperature: float,
) -> Tuple[PairBarResult, List[Trajectory]]:
    """Sequentially run simulations at each state in initial_states,
    returning summaries that can be used for pair BAR, energy decomposition, and other diagnostics

    Returns
    -------
    PairBarResult
        Results of pair BAR analysis

    list of Trajectory
        Trajectory for each state in initial_states

    Notes
    -----
    * Memory complexity:
        Memory demand should be no more than that of 2 states worth of frames.
        Disk demand is proportional to number of initial states.

        This restriction may need to be relaxed in the future if:
        * We decide to use MBAR(states) rather than sum_i BAR(states[i], states[i+1])
        * We use online protocol optimization approaches that require more states to be kept on-hand
    """
    stored_trajectories = []

    # keep no more than 2 states in memory at once
    prev_state: Optional[EnergyDecomposedState] = None

    # u_kln matrix (2, 2, n_frames) for each pair of adjacent lambda windows and energy term
    u_kln_by_component_by_lambda = []

    # NOTE: this assumes that states differ only in their parameters, but we do not check this!
    unbound_impls = [p.potential.to_gpu(np.float32).unbound_impl for p in initial_states[0].potentials]
    for initial_state in initial_states:
        # run simulation
        traj = sample(initial_state, md_params, max_buffer_frames=100)
        print(f"completed simulation at lambda={initial_state.lamb}!")

        # keep samples from any requested states in memory
        stored_trajectories.append(traj)

        cur_batch_U_fns = get_batch_u_fns(unbound_impls, [p.params for p in initial_state.potentials], temperature)

        state = EnergyDecomposedState(traj.frames, traj.boxes, cur_batch_U_fns)

        # analysis that depends on current and previous state
        if prev_state:
            state_pair = [prev_state, state]
            u_kln_by_component = compute_energy_decomposed_u_kln(state_pair)
            u_kln_by_component_by_lambda.append(u_kln_by_component)

        prev_state = state

    bar_results = [
        estimate_free_energy_bar(u_kln_by_component, temperature) for u_kln_by_component in u_kln_by_component_by_lambda
    ]

    return PairBarResult(list(initial_states), bar_results), stored_trajectories


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
) -> Tuple[List[PairBarResult], List[Trajectory]]:
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
    list of IntermediateResult
        For each iteration of bisection, object containing the current list of states and array of energy-decomposed
        u_kln matrices.

    list of Trajectory
        Trajectory for each state
    """

    assert len(initial_lambdas) >= 2
    assert np.all(np.diff(initial_lambdas) > 0), "initial lambda schedule must be monotonically increasing"

    lambdas = list(initial_lambdas)

    get_initial_state = cache(make_initial_state)

    @cache
    def get_samples(lamb: float) -> Trajectory:
        initial_state = get_initial_state(lamb)
        traj = sample(initial_state, md_params, max_buffer_frames=100)
        return traj

    # Set up a single set of unbound potentials for computing the batch U fns
    # NOTE: this assumes that states differ only in their parameters, but we do not check this!
    unbound_impls = [p.potential.to_gpu(np.float32).unbound_impl for p in get_initial_state(lambdas[0]).potentials]

    # NOTE: we don't cache get_state to avoid holding BoundPotentials in memory since they
    # 1. can use significant GPU memory
    # 2. can be reconstructed relatively quickly
    def get_state(lamb: float) -> EnergyDecomposedState[StoredArrays]:
        initial_state = get_initial_state(lamb)
        traj = get_samples(lamb)
        batch_u_fns = get_batch_u_fns(unbound_impls, [p.params for p in initial_state.potentials], temperature)
        return EnergyDecomposedState(traj.frames, traj.boxes, batch_u_fns)

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

            if min_overlap is not None:
                overlap_info = f"Current minimum BAR overlap {cost_to_overlap(max(costs)):.3g} <= {min_overlap:.3g} "
            else:
                overlap_info = f"Current minimum BAR overlap {cost_to_overlap(max(costs)):.3g} (min_overlap == None) "

            print(
                f"Bisection iteration {iteration} (of {n_bisections}): "
                + overlap_info
                + f"between states at λ={lamb1:.3g} and λ={lamb2:.3g}. "
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

    trajectories = [get_samples(lamb) for lamb in lambdas]

    return results, trajectories


def run_sims_hrex(
    initial_states: Sequence[InitialState],
    md_params: MDParams,
    n_frames_per_iter: int,
    n_swap_attempts_per_iter: Optional[int] = None,
    print_diagnostics_interval: Optional[int] = 10,
) -> Tuple[PairBarResult, List[Trajectory], HREXDiagnostics]:
    r"""Sample from a sequence of states using nearest-neighbor Hamiltonian Replica EXchange (HREX).

    See documentation for :py:func:`timemachine.md.hrex.run_hrex` for details of the algorithm and implementation.

    Parameters
    ----------
    initial_states: sequence of InitialState
        States to sample. Should be ordered such that adjacent states have significant overlap for good mixing
        performance

    md_params: MDParams
        MD parameters

    n_frames_per_iter: int
        Number of frames to sample using MD per iteration

    n_swap_attempts_per_iter: int or None, optional
        Number of nearest-neighbor swaps to attempt per iteration. Defaults to len(initial_states) ** 4.

    print_diagnostics_interval: int or None, optional
        If not None, print diagnostics every N iterations

    Returns
    -------
    PairBarResult
        results of pair BAR free energy analysis

    list of Trajectory
        Trajectory for each state

    HREXDiagnostics
        HREX statistics (e.g. swap rates, replica-state distribution)
    """

    # TODO: to support replica exchange with variable temperatures,
    #  consider modifying sample fxn to rescale velocities by sqrt(T_new/T_orig)
    def assert_ensembles_compatible(state_a: InitialState, state_b: InitialState):
        """check that xvb from state_a can be swapped with xvb from state_b (up to timestep error),
        with swap acceptance probability that depends only on U_a, U_b, kBT"""

        # assert (A, B) have identical masses, temperature
        intg_a = state_a.integrator
        intg_b = state_b.integrator

        assert (intg_a.masses == intg_b.masses).all()
        assert intg_a.temperature == intg_b.temperature

        # assert same pressure (or same volume)
        assert (state_a.barostat is None) == (state_b.barostat is None), "should both be NVT or both be NPT"

        if state_a.barostat and state_b.barostat:
            # assert (A, B) are compatible NPT ensembles
            baro_a: MonteCarloBarostat = state_a.barostat
            baro_b: MonteCarloBarostat = state_b.barostat

            assert baro_a.pressure == baro_b.pressure
            assert baro_a.temperature == baro_b.temperature

            # also, assert barostat and integrator are self-consistent
            assert intg_a.temperature == baro_a.temperature

        else:
            # assert (A, B) are compatible NVT ensembles
            assert (state_a.box0 == state_b.box0).all()

    for s in initial_states[1:]:
        assert_ensembles_compatible(initial_states[0], s)

    if n_swap_attempts_per_iter is None:
        n_swap_attempts_per_iter = get_swap_attempts_per_iter_heuristic(len(initial_states))

    # Set up overall potential and context using the first state.
    # NOTE: this assumes that states differ only in their parameters, but we do not check this!
    context = get_context(initial_states[0], md_params=md_params)
    bound_potentials = context.get_potentials()
    assert len(bound_potentials) == 1
    potential = bound_potentials[0].get_potential()
    temperature = initial_states[0].integrator.temperature
    ligand_idxs = initial_states[0].ligand_idxs

    def get_flattened_params(initial_state: InitialState) -> NDArray:
        return np.concatenate([bp.params.flatten() for bp in initial_state.potentials])

    params_by_state = np.array([get_flattened_params(initial_state) for initial_state in initial_states])
    water_params_by_state: Optional[NDArray] = None
    if md_params.water_sampling_params is not None:
        water_params_by_state = np.array([get_water_sampler_params(initial_state) for initial_state in initial_states])

    def compute_log_q_matrix(xvbs: List[CoordsVelBox]) -> NDArray:
        coords = np.array([xvb.coords for xvb in xvbs])
        boxes = np.array([xvb.box for xvb in xvbs])
        _, _, U = potential.execute_batch(coords, params_by_state, boxes, False, False, True)
        log_q = -U / (BOLTZ * temperature)
        return log_q

    state_idxs = [StateIdx(i) for i, _ in enumerate(initial_states)]
    neighbor_pairs = list(zip(state_idxs, state_idxs[1:]))

    if len(initial_states) == 2:
        # Add an identity move to the mixture to ensure aperiodicity
        neighbor_pairs = [(StateIdx(0), StateIdx(0))] + neighbor_pairs

    # Setup the barostat to run more often for equilibration
    equil_barostat_interval = 15
    barostat = context.get_barostat()
    if barostat is not None:
        barostat.set_interval(equil_barostat_interval)

    def get_equilibrated_xvb(xvb: CoordsVelBox, state_idx: StateIdx) -> CoordsVelBox:
        # Set up parameters of the water sampler movers
        if md_params.water_sampling_params is not None:
            for mover in context.get_movers():
                if isinstance(mover, WATER_SAMPLER_MOVERS):
                    assert water_params_by_state is not None
                    mover.set_params(water_params_by_state[state_idx])
        if md_params.n_eq_steps > 0:
            # Set the movers to 0 to ensure they all equilibrate the same way
            # Ensure initial mover state is consistent across replicas
            for mover in context.get_movers():
                mover.set_step(0)

            params = params_by_state[state_idx]
            bound_potentials[0].set_params(params)

            context.set_x_t(xvb.coords)
            context.set_v_t(xvb.velocities)
            context.set_box(xvb.box)

            xs, boxes = context.multiple_steps(md_params.n_eq_steps, store_x_interval=0)
            assert len(xs) == 1
            x0 = xs[0]
            v0 = context.get_v_t()
            box0 = boxes[0]
            xvb = CoordsVelBox(x0, v0, box0)
        return xvb

    initial_replicas = [
        get_equilibrated_xvb(CoordsVelBox(s.x0, s.v0, s.box0), StateIdx(i)) for i, s in enumerate(initial_states)
    ]

    hrex = HREX.from_replicas(initial_replicas)

    samples_by_state: List[Trajectory] = [Trajectory.empty() for _ in initial_states]
    replica_idx_by_state_by_iter: List[List[ReplicaIdx]] = []
    fraction_accepted_by_pair_by_iter: List[List[Tuple[int, int]]] = []

    # Reset the barostat from the equilibration interval to the production interval
    barostat = context.get_barostat()
    state = initial_states[0]
    if barostat is not None and state.barostat is not None:
        barostat.set_interval(state.barostat.interval)

    if (
        md_params.water_sampling_params is not None
        and md_params.steps_per_frame * n_frames_per_iter > md_params.water_sampling_params.interval
    ):
        warn("Not running any water sampling, too few steps of MD for the water sampling interval")

    for iteration, n_frames_iter in enumerate(batches(md_params.n_frames, n_frames_per_iter), 1):

        def sample_replica(xvb: CoordsVelBox, state_idx: StateIdx) -> Trajectory:
            context.set_x_t(xvb.coords)
            context.set_v_t(xvb.velocities)
            context.set_box(xvb.box)

            params = params_by_state[state_idx]
            bound_potentials[0].set_params(params)

            current_step = (iteration - 1) * n_frames_per_iter * md_params.steps_per_frame
            # Setup the MC movers of the Context
            for mover in context.get_movers():
                if md_params.water_sampling_params is not None and isinstance(mover, WATER_SAMPLER_MOVERS):
                    assert water_params_by_state is not None
                    mover.set_params(water_params_by_state[state_idx])
                # Set the step so that all windows have the movers be called the same number of times.
                mover.set_step(current_step)

            md_params_replica = replace(
                md_params, n_frames=n_frames_iter, n_eq_steps=0, seed=np.random.randint(np.iinfo(np.int32).max)
            )

            return sample_with_context(context, md_params_replica, temperature, ligand_idxs, max_buffer_frames=100)

        def replica_from_samples(traj: Trajectory) -> CoordsVelBox:
            return CoordsVelBox(traj.frames[-1], traj.final_velocities, traj.boxes[-1])

        hrex, samples_by_state_iter = hrex.sample_replicas(sample_replica, replica_from_samples)
        log_q_kl = compute_log_q_matrix(hrex.replicas)
        hrex, fraction_accepted_by_pair = hrex.attempt_neighbor_swaps_fast(
            neighbor_pairs, log_q_kl, n_swap_attempts_per_iter, md_params.seed + iteration
        )

        if len(initial_states) == 2:
            fraction_accepted_by_pair = fraction_accepted_by_pair[1:]  # remove stats for identity move

        for samples, samples_iter in zip(samples_by_state, samples_by_state_iter):
            samples.extend(samples_iter)

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

    # Use the unbound potentials associated with the summed potential once to compute the u_kln
    # Avoids repeated creation of underlying GPU potentials
    assert isinstance(potential, custom_ops.SummedPotential)
    unbound_impls = potential.get_potentials()

    def make_energy_decomposed_state(
        results: Tuple[StoredArrays, List[NDArray], InitialState]
    ) -> EnergyDecomposedState[StoredArrays]:
        frames, boxes, initial_state = results
        # Reuse the existing unbound potentials already constructed to make a batch Us fn
        return EnergyDecomposedState(
            frames,
            boxes,
            get_batch_u_fns(unbound_impls, [p.params for p in initial_state.potentials], temperature),
        )

    results_by_state = [
        (samples.frames, samples.boxes, initial_state)
        for samples, initial_state in zip(samples_by_state, initial_states)
    ]

    bar_results = list(
        pairwise_transform_and_combine(
            results_by_state,
            make_energy_decomposed_state,
            lambda s1, s2: estimate_free_energy_bar(compute_energy_decomposed_u_kln([s1, s2]), temperature),
        )
    )

    diagnostics = HREXDiagnostics(replica_idx_by_state_by_iter, fraction_accepted_by_pair_by_iter)

    return PairBarResult(list(initial_states), bar_results), samples_by_state, diagnostics
