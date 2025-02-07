"""Absolute hydration free energies"""

import pickle
from collections.abc import Sequence
from functools import partial

import numpy as np
from numpy.typing import NDArray as Array

from timemachine import potentials
from timemachine.constants import BOLTZ, DEFAULT_TEMP
from timemachine.fe import model_utils
from timemachine.fe.free_energy import (
    AbsoluteFreeEnergy,
    HostConfig,
    InitialState,
    MDParams,
    SimulationResult,
    make_pair_bar_plots,
    run_sims_sequential,
)
from timemachine.fe.lambda_schedule import construct_pre_optimized_absolute_lambda_schedule_solvent
from timemachine.fe.topology import BaseTopology
from timemachine.fe.utils import get_mol_name, get_romol_conf
from timemachine.ff import Forcefield
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat
from timemachine.md import builders, enhanced, minimizer, smc
from timemachine.md.barostat.moves import NPTMove
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.states import CoordsVelBox
from timemachine.potentials import SummedPotential
from timemachine.potentials.potential import get_potential_by_type

DEFAULT_AHFE_MD_PARAMS = MDParams(n_frames=1000, n_eq_steps=10_000, steps_per_frame=400, seed=2023)


def generate_endstate_samples(
    num_samples: int,
    solvent_samples: Sequence[CoordsVelBox],
    ligand_samples: Sequence,
    ligand_log_weights: Array,
    num_ligand_atoms: int,
) -> list[CoordsVelBox]:
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
        noninteracting_xvb = solvent_samples[choice_idx]  # type: ignore[call-overload]

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
    if not isinstance(seed, int):
        seed = np.random.randint(1000)
        print(f"setting seed randomly to {seed}")
    else:
        print(f"setting seed to {seed}")

    np.random.seed(seed)

    # set up potentials
    ff = ff or Forcefield.load_default()
    potentials, params, masses, _, _ = enhanced.get_solvent_phase_system(mol, ff)

    U_fn = SummedPotential(potentials, params)
    kBT = BOLTZ * temperature

    def reduced_potential_fxn(xvb, lam):
        return U_fn(xvb.coords, params, xvb.box, lam) / kBT

    for U, p in zip(potentials, params):
        U.bind(p)

    npt_mover = NPTMove(potentials, None, masses, temperature, pressure, n_steps=n_steps, seed=seed)

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
    prefix="",
    md_params: MDParams = DEFAULT_AHFE_MD_PARAMS,
    n_windows=None,
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

    prefix: str
        A prefix to append to figures

    n_windows: None
        Number of windows used for interpolating the the lambda schedule with additional windows.

    md_params: MDParams
        Parameters for the equilibration and production MD. Defaults to 400 global steps per frame, 1000 frames and 10k
        equilibration steps with seed 2023.

    Returns
    -------
    SimulationResult
        Collected data from the simulation (see class for storage information). We currently return frames
        from only the first and last window.
    """
    bt = BaseTopology(mol, ff)
    afe = AbsoluteFreeEnergy(mol, bt)
    if md_params is None:
        md_params = MDParams(n_frames=2000, steps_per_frame=400, n_eq_steps=200000)

    # note: tm convention lambda=1 means "decoupled", lambda=0 means "coupled"
    lambda_schedule = construct_pre_optimized_absolute_lambda_schedule_solvent(n_windows)[::-1]
    assert np.isclose(lambda_schedule[0], 1.0) and np.isclose(lambda_schedule[-1], 0.0)

    temperature = DEFAULT_TEMP
    initial_states = setup_initial_states(afe, ff, host_config, temperature, lambda_schedule, md_params.seed)

    combined_prefix = get_mol_name(mol) + "_" + prefix
    try:
        result, stored_trajectories = run_sims_sequential(initial_states, md_params, temperature)
        plots = make_pair_bar_plots(result, temperature, combined_prefix)
        return SimulationResult(result, plots, stored_trajectories, md_params, [])
    except Exception as err:
        with open(f"failed_ahfe_result_{combined_prefix}.pkl", "wb") as fh:
            pickle.dump((initial_states, md_params, err), fh)
        raise err


def setup_initial_states(
    afe: AbsoluteFreeEnergy,
    ff: Forcefield,
    host_config: HostConfig,
    temperature: float,
    lambda_schedule: Array,
    seed: int,
) -> list[InitialState]:
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

    host_conf = minimizer.fire_minimize_host(
        [afe.mol],
        host_config,
        ff,
    )

    initial_states = []

    # check that the lambda schedule is monotonically decreasing.
    assert np.all(np.diff(lambda_schedule) < 0)

    for _, lamb in enumerate(lambda_schedule):
        ligand_conf = get_romol_conf(afe.mol)

        ubps, params, masses = afe.prepare_host_edge(ff, host_config, lamb)
        x0 = afe.prepare_combined_coords(host_coords=host_conf)
        bps = []
        for ubp, param in zip(ubps, params):
            bp = ubp.bind(param)
            bps.append(bp)

        bond_potential = get_potential_by_type(ubps, potentials.HarmonicBond)

        hmr_masses = model_utils.apply_hmr(masses, bond_potential.idxs)
        group_idxs = get_group_indices(get_bond_list(bond_potential), len(masses))
        baro = MonteCarloBarostat(len(hmr_masses), 1.0, temperature, group_idxs, 15, seed)
        box0 = host_config.box

        v0 = np.zeros_like(x0)  # tbd resample from Maxwell-boltzman?
        num_ligand_atoms = len(ligand_conf)
        num_total_atoms = len(x0)
        ligand_idxs = np.arange(num_total_atoms - num_ligand_atoms, num_total_atoms)

        dt = 2.5e-3
        friction = 1.0
        intg = LangevinIntegrator(temperature, dt, friction, hmr_masses, seed)

        state = InitialState(bps, intg, baro, x0, v0, box0, lamb, ligand_idxs, np.array([], dtype=np.int32))
        initial_states.append(state)
    return initial_states


def run_solvent(
    mol, forcefield: Forcefield, _, md_params: MDParams, n_windows=16
) -> tuple[SimulationResult, HostConfig]:
    box_width = 4.0
    solvent_host_config = builders.build_water_system(box_width, forcefield.water_ff, mols=[mol])
    solvent_host_config.box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes, deboggle later
    solvent_res = estimate_absolute_free_energy(
        mol,
        forcefield,
        solvent_host_config,
        md_params=md_params,
        prefix="solvent",
        n_windows=n_windows,
    )
    return solvent_res, solvent_host_config
