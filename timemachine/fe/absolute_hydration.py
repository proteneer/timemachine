"""Absolute hydration free energies"""

from functools import partial

import numpy as np
from tqdm import tqdm

from timemachine.constants import BOLTZ
from timemachine.fe import free_energy
from timemachine.fe.free_energy_rabfe import construct_pre_optimized_absolute_lambda_schedule_solvent
from timemachine.md import enhanced, builders, moves
from timemachine.md.smc import conditional_multinomial_resample
from timemachine.md.states import CoordsVelBox
from timemachine.utils import get_ff_am1ccc, construct_potential, bind_potentials


def generate_endstate_samples(num_samples, solvent_samples, ligand_samples, ligand_log_weights, num_ligand_atoms):
    """solvent + (noninteracting ligand) sample --> solvent + (vacuum ligand) sample

    Assumptions:
    ------------
    * ligand indices: last num_ligand_atoms

    TODO: document me more"""
    all_xvbs = []
    for _ in tqdm(range(num_samples), desc="generating endstate samples"):
        choice_idx = np.random.choice(np.arange(len(solvent_samples)))
        solvent_x = solvent_samples[choice_idx].coords
        solvent_v = solvent_samples[choice_idx].velocities
        ligand_xv = enhanced.sample_from_log_weights(ligand_samples, ligand_log_weights, size=1)[0]
        ligand_x = ligand_xv[0]
        ligand_v = ligand_xv[1]
        combined_x = np.concatenate([solvent_x[:-num_ligand_atoms], ligand_x], axis=0)
        combined_v = np.concatenate([solvent_v[:-num_ligand_atoms], ligand_v], axis=0)
        combined_box = solvent_samples[choice_idx].box
        all_xvbs.append(CoordsVelBox(combined_x, combined_v, combined_box))
    return all_xvbs


def get_solvent_phase_system(mol, ff):
    water_system, water_coords, water_box, water_topology = builders.build_water_system(3.0)
    water_box = water_box + np.eye(3) * 0.5  # add a small margin around the box for stability
    afe = free_energy.AbsoluteFreeEnergy(mol, ff)
    ff_params = ff.get_ordered_params()
    ubps, params, masses, coords = afe.prepare_host_edge(ff_params, water_system, water_coords)
    return ubps, params, masses, coords, water_box


def setup_absolute_hydration_with_endpoint_samples(mol, temperature=300.0, pressure=1.0, n_steps=1000):
    """Generate samples from the equilibrium distribution at lambda=1

    Return:
    * reduced_potential_fxn
    * npt_mover
    * initial_samples from lam = 1
    """

    seed = 2022
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


def set_up_ahfe_system_for_smc(mol, n_walkers, n_windows, n_md_steps, resample_thresh):
    """define initial samples, lambdas schedule, propagate fxn, log_prob fxn, resample fxn"""
    reduced_potential, mover, initial_samples = setup_absolute_hydration_with_endpoint_samples(mol, n_steps=n_md_steps)

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
