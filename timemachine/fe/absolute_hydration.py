"""Absolute hydration free energies"""

from functools import partial

import numpy as np
from tqdm import tqdm

from timemachine.constants import BOLTZ
from timemachine.fe import free_energy
from timemachine.fe.lambda_schedule import construct_pre_optimized_absolute_lambda_schedule_solvent
from timemachine.md import builders, enhanced, moves
from timemachine.md.smc import conditional_multinomial_resample
from timemachine.md.states import CoordsVelBox
from timemachine.utils import bind_potentials, construct_potential, get_ff_am1ccc


def generate_endstate_samples(num_samples, solvent_samples, ligand_samples, ligand_log_weights, num_ligand_atoms):
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
    * list of (coordinates, velocities, box), distributed according to


    Assumptions
    -----------
    * ligand indices: last num_ligand_atoms

    Notes
    -----
    * TODO[generality]: refactor to accept two streams of unweighted samples, concatenate them
        (rather than requiring and discarding ligand component in solvent_samples, performing multinomial resampling)
    """

    # assume this layout
    num_total_atoms = len(solvent_samples[0].coords)
    num_solvent_atoms = num_total_atoms - num_ligand_atoms
    solvent_idxs = np.arange(0, num_solvent_atoms)

    # sample according to log weights
    ligand_xvs = enhanced.sample_from_log_weights(ligand_samples, ligand_log_weights, size=num_samples)

    # sample uniformly with replacement
    solvent_choice_idxs = np.random.choice(np.arange(len(solvent_samples)), size=num_samples)

    all_xvbs = []
    for i, choice_idx in enumerate(tqdm(solvent_choice_idxs, desc="generating endstate samples")):

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


def get_solvent_phase_system(mol, ff):
    water_system, water_coords, water_box, water_topology = builders.build_water_system(3.0)
    water_box = water_box + np.eye(3) * 0.5  # add a small margin around the box for stability
    afe = free_energy.AbsoluteFreeEnergy(mol, ff)
    ff_params = ff.get_ordered_params()
    ubps, params, masses, coords = afe.prepare_host_edge(ff_params, water_system, water_coords)
    return ubps, params, masses, coords, water_box


def setup_absolute_hydration_with_endpoint_samples(mol, temperature=300.0, pressure=1.0, n_steps=1000, seed=2022):
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
    ff = get_ff_am1ccc()
    ubps, params, masses, _, _ = enhanced.get_solvent_phase_system(mol, ff)
    potential_fxn = construct_potential(ubps, params)

    kBT = BOLTZ * temperature

    def reduced_potential_fxn(xvb, lam):
        return potential_fxn(xvb, lam) / kBT

    bind_potentials(ubps, params)

    # set up npt mover
    npt_mover = moves.NPTMove(ubps, None, masses, temperature, pressure, n_steps, seed)

    # combine solvent and ligand samples
    solvent_xvbs, ligand_samples, ligand_log_weights = enhanced.load_or_pregenerate_samples(
        mol, ff, seed, temperature=temperature, pressure=pressure
    )
    n_endstate_samples = 5000  # TODO: expose this parameter?
    num_ligand_atoms = mol.GetNumAtoms()
    all_xvbs = generate_endstate_samples(
        n_endstate_samples, solvent_xvbs, ligand_samples, ligand_log_weights, num_ligand_atoms
    )

    return reduced_potential_fxn, npt_mover, all_xvbs


def set_up_ahfe_system_for_smc(mol, n_walkers, n_windows, n_md_steps, resample_thresh, seed=2022):
    """define initial samples, lambdas schedule, propagate fxn, log_prob fxn, resample fxn"""
    reduced_potential, mover, initial_samples = setup_absolute_hydration_with_endpoint_samples(
        mol, n_steps=n_md_steps, seed=seed
    )

    sample_inds = np.random.choice(np.arange(len(initial_samples)), size=n_walkers)
    samples = [initial_samples[i] for i in sample_inds]

    lambdas = construct_pre_optimized_absolute_lambda_schedule_solvent(n_windows)[::-1]

    def propagate(xs, lam):
        mover.lamb = lam
        xs_next = [mover.move(x) for x in xs]
        return xs_next

    def log_prob(xs, lam):
        u_s = np.array([reduced_potential(x, lam) for x in xs])
        return -u_s

    resample = partial(conditional_multinomial_resample, thresh=resample_thresh)

    return samples, lambdas, propagate, log_prob, resample
