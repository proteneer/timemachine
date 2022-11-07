"""Absolute hydration free energies"""

import pickle
from functools import partial
from typing import List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray as Array
from simtk.openmm import app

from timemachine.constants import BOLTZ, DEFAULT_FF, DEFAULT_TEMP
from timemachine.fe import functional, model_utils
from timemachine.fe.free_energy import (
    AbsoluteFreeEnergy,
    HostConfig,
    InitialState,
    SimulationProtocol,
    SimulationResult,
)
from timemachine.fe.lambda_schedule import construct_pre_optimized_absolute_lambda_schedule_solvent
from timemachine.fe.rbfe import estimate_free_energy_given_initial_states
from timemachine.fe.topology import BaseTopology
from timemachine.fe.utils import get_mol_name, get_romol_conf
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, potentials
from timemachine.md import builders, enhanced, minimizer, moves, smc
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.states import CoordsVelBox


def generate_endstate_samples(
    num_samples: int,
    solvent_samples: Sequence[CoordsVelBox],
    ligand_samples: Sequence,
    ligand_log_weights: Array,
    num_ligand_atoms: int,
) -> List[CoordsVelBox]:
    """solvent + (noninteracting ligand) sample --> solvent + (vacuum ligand) sample

    Inputs
    ------
    * solvent_samples ~ p_noninteracting([x_solvent, x_ligand])
    * (ligand_samples, ligand_log_weights) ~ p_vacuum(x_ligand)
        (note: this set of samples is importance-weighted!)

    Processing
    ----------
    * resample ligand_samples according to ligand_log_weights
    * concatenate solvent component from p_noninteracting with ligand from p_vacuum

    Returns
    -------
    * list of (coordinates, velocities, box), distributed according to p_noninteracting

    Assumptions
    -----------
    * ligand indices: last num_ligand_atoms

    Notes
    -----
    * TODO[generality]: refactor to accept two streams of unweighted samples, concatenate them
        (rather than requiring and discarding ligand component in solvent_samples, performing multinomial resampling)
    * TODO[logic]: generate solvent samples once independently of ligand
        (rather than duplicating work of solvent sampling across all ligands)
    """

    # assume this layout
    num_total_atoms = len(solvent_samples[0].coords)
    num_solvent_atoms = num_total_atoms - num_ligand_atoms
    assert num_solvent_atoms > 0, "Oops, did you really mean num_ligand_atoms >= num_total_atoms?"
    solvent_idxs = np.arange(0, num_solvent_atoms)

    # sample according to log weights
    ligand_xvs = enhanced.sample_from_log_weights(ligand_samples, ligand_log_weights, size=num_samples)

    # sample uniformly with replacement
    solvent_choice_idxs = np.random.choice(len(solvent_samples), size=num_samples, replace=True)

    all_xvbs = []
    for i, choice_idx in enumerate(solvent_choice_idxs):

        # solvent + noninteracting ligand
        noninteracting_xvb = solvent_samples[choice_idx]

        # vacuum ligand
        ligand_x, ligand_v = ligand_xvs[i]

        # concatenate solvent
        combined_x = np.concatenate([noninteracting_xvb.coords[solvent_idxs], ligand_x], axis=0)
        combined_v = np.concatenate([noninteracting_xvb.velocities[solvent_idxs], ligand_v], axis=0)

        combined_xvb = CoordsVelBox(combined_x, combined_v, noninteracting_xvb.box)

        all_xvbs.append(combined_xvb)
    return all_xvbs


def setup_absolute_hydration_with_endpoint_samples(
    mol, temperature=300.0, pressure=1.0, n_steps=1000, seed=2022, ff=None, num_workers=None
):
    """Generate samples from the equilibrium distribution at lambda=1

    Return:
    * reduced_potential_fxn
    * npt_mover
    * initial_samples from lam = 1
    """
    if type(seed) != int:
        seed = np.random.randint(1000)
        print(f"setting seed randomly to {seed}")
    else:
        print(f"setting seed to {seed}")

    np.random.seed(seed)

    # set up potentials
    ff = ff or Forcefield.load_from_file(DEFAULT_FF)
    potentials, params, masses, _, _ = enhanced.get_solvent_phase_system(mol, ff)

    U_fn = functional.construct_differentiable_interface_fast(potentials, params)
    kBT = BOLTZ * temperature

    def reduced_potential_fxn(xvb, lam):
        return U_fn(xvb.coords, params, xvb.box, lam) / kBT

    for U, p in zip(potentials, params):
        U.bind(p)

    npt_mover = moves.NPTMove(potentials, None, masses, temperature, pressure, n_steps, seed)

    # combine solvent and ligand samples
    solvent_xvbs, ligand_samples, ligand_log_weights = enhanced.pregenerate_samples(
        mol, ff, seed, temperature=temperature, pressure=pressure, num_workers=num_workers
    )
    n_endstate_samples = 5000  # TODO: expose this parameter?
    num_ligand_atoms = mol.GetNumAtoms()
    all_xvbs = generate_endstate_samples(
        n_endstate_samples, solvent_xvbs, ligand_samples, ligand_log_weights, num_ligand_atoms
    )

    return reduced_potential_fxn, npt_mover, all_xvbs


def set_up_ahfe_system_for_smc(
    mol, n_walkers, n_windows, n_md_steps, resample_thresh, seed=2022, ff=None, num_workers=None
):
    """define initial samples, lambdas schedule, propagate fxn, log_prob fxn, resample fxn"""
    reduced_potential, mover, initial_samples = setup_absolute_hydration_with_endpoint_samples(
        mol, n_steps=n_md_steps, seed=seed, ff=ff, num_workers=num_workers
    )
    np.random.seed(seed)

    sample_inds = np.random.choice(np.arange(len(initial_samples)), size=n_walkers, replace=True)
    samples = [initial_samples[i] for i in sample_inds]

    # note: tm convention lambda=1 means "decoupled", lambda=0 means "coupled"
    lambdas = construct_pre_optimized_absolute_lambda_schedule_solvent(n_windows)

    def propagate(xs, lam):
        mover.lamb = lam
        xs_next = [mover.move(x) for x in xs]
        return xs_next

    def log_prob(xs, lam):
        u_s = np.array([reduced_potential(x, lam) for x in xs])
        return -u_s

    resample = partial(smc.conditional_multinomial_resample, thresh=resample_thresh)

    return samples, lambdas, propagate, log_prob, resample


def estimate_absolute_free_energy(
    mol,
    ff: Forcefield,
    host_config: HostConfig,
    seed: int,
    n_frames=1000,
    prefix="",
    n_windows=None,
    keep_idxs=None,
    n_eq_steps=10000,
    steps_per_frame=400,
    image_traj=True,
):
    """
    Estimate the absolute hydration free energy for the given mol.

    Parameters
    ----------
    mol: Chem.Mol
        molecule

    ff: ff.Forcefield
        Forcefield to be used for the system

    host_config: HostConfig
        Configuration for the host system.

    n_frames: int
        number of samples to generate for each lambda window, where each sample is `steps_per_frame` steps of MD.

    prefix: str
        A prefix to append to figures

    seed: int
        Random seed to use for the simulations.

    n_windows: None
        Number of windows used for interpolating the the lambda schedule with additional windows.

    keep_idxs: list of int or None
        If None, return only the end-state frames. Otherwise if not None, use only for debugging, and this
        will return the frames corresponding to the idxs of interest.

    n_eq_steps: int
        Number of equilibration steps for each window.

    steps_per_frame: int
        The number of steps to take before collecting a frame

    image_traj: bool
        Images the trajectories returned with the SimulationResult. Recenters
        frames around the ligand and then wraps the coordinates into the box.

    Returns
    -------
    SimulationResult
        Collected data from the simulation (see class for storage information). We currently return frames
        from only the first and last window.
    """
    bt = BaseTopology(mol, ff)
    afe = AbsoluteFreeEnergy(mol, bt)

    # note: tm convention lambda=1 means "decoupled", lambda=0 means "coupled"
    lambda_schedule = construct_pre_optimized_absolute_lambda_schedule_solvent(n_windows)[::-1]
    assert np.isclose(lambda_schedule[0], 1.0) and np.isclose(lambda_schedule[-1], 0.0)

    temperature = DEFAULT_TEMP
    initial_states = setup_initial_states(afe, ff, host_config, temperature, lambda_schedule, seed)
    protocol = SimulationProtocol(n_frames=n_frames, n_eq_steps=n_eq_steps, steps_per_frame=steps_per_frame)

    if keep_idxs is None:
        keep_idxs = [0, len(initial_states) - 1]  # keep first and last windows
    assert len(keep_idxs) <= len(lambda_schedule)

    combined_prefix = get_mol_name(mol) + "_" + prefix
    try:
        return estimate_free_energy_given_initial_states(
            initial_states, protocol, temperature, combined_prefix, keep_idxs, image_traj=image_traj
        )
    except Exception as err:
        with open(f"failed_ahfe_result_{combined_prefix}.pkl", "wb") as fh:
            pickle.dump((initial_states, protocol, err), fh)
        raise err


def setup_initial_states(
    afe: AbsoluteFreeEnergy,
    ff: Forcefield,
    host_config: HostConfig,
    temperature: float,
    lambda_schedule: Array,
    seed: int,
) -> List[InitialState]:
    """
    Setup the initial states for a series of lambda values. It is assumed that the lambda schedule
    is a monotonically decreasing sequence in the closed interval [0, 1].

    Parameters
    ----------
    afe: AbsoluteFreeEnergy
        An AbsoluteFreeEnergy object which contains the mol structure

    ff: ff.Forcefield
        Forcefield to be used for the system

    host_config: HostConfig
        Configurations of the host.

    temperature: float
        Temperature to run the simulation at.

    lambda_schedule: list of float

    seed: int
        Random number seed

    Returns
    -------
    list of InitialStates
        Returns an initial state for each value of lambda.

    """
    host_bps, host_masses = openmm_deserializer.deserialize_system(host_config.omm_system, cutoff=1.2)
    host_conf = minimizer.minimize_host_4d(
        [afe.mol],
        host_config.omm_system,
        host_config.conf,
        ff,
        host_config.box,
    )

    initial_states = []

    # check that the lambda schedule is monotonically decreasing.
    assert np.all(np.diff(lambda_schedule) < 0)

    for lamb_idx, lamb in enumerate(lambda_schedule):
        ligand_conf = get_romol_conf(afe.mol)

        ubps, params, masses = afe.prepare_host_edge(ff.get_params(), host_config.omm_system, lamb)
        x0 = afe.prepare_combined_coords(host_coords=host_conf)
        bps = []
        for ubp, param in zip(ubps, params):
            bp = ubp.bind(param)
            bps.append(bp)

        bond_potential = ubps[0]
        assert isinstance(bond_potential, potentials.HarmonicBond)
        hmr_masses = model_utils.apply_hmr(masses, bond_potential.get_idxs())
        group_idxs = get_group_indices(get_bond_list(bond_potential))
        baro = MonteCarloBarostat(len(hmr_masses), 1.0, temperature, group_idxs, 15, seed)
        box0 = host_config.box

        v0 = np.zeros_like(x0)  # tbd resample from Maxwell-boltzman?
        num_ligand_atoms = len(ligand_conf)
        num_total_atoms = len(x0)
        ligand_idxs = np.arange(num_total_atoms - num_ligand_atoms, num_total_atoms)

        dt = 2.5e-3
        friction = 1.0
        intg = LangevinIntegrator(temperature, dt, friction, hmr_masses, seed)

        state = InitialState(bps, intg, baro, x0, v0, box0, lamb, ligand_idxs)
        initial_states.append(state)
    return initial_states


def run_solvent(
    mol, forcefield, _, n_frames, seed, n_eq_steps=10000, steps_per_frame=400, n_windows=16
) -> Tuple[SimulationResult, app.topology.Topology]:
    box_width = 4.0
    solvent_sys, solvent_conf, solvent_box, solvent_top = builders.build_water_system(box_width, forcefield.water_ff)
    solvent_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes, deboggle later
    solvent_host_config = HostConfig(solvent_sys, solvent_conf, solvent_box)
    solvent_res = estimate_absolute_free_energy(
        mol,
        forcefield,
        solvent_host_config,
        seed,
        prefix="solvent",
        n_frames=n_frames,
        n_eq_steps=n_eq_steps,
        n_windows=n_windows,
        steps_per_frame=steps_per_frame,
    )
    return solvent_res, solvent_top
