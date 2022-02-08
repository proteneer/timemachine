"""Absolute hydration free energies"""

import numpy as np
from tqdm import tqdm

from timemachine.constants import BOLTZ
from timemachine.fe import free_energy
from timemachine.md import enhanced, builders, moves
from timemachine.md.states import CoordsVelBox
from timemachine.utils import get_ff_am1ccc, construct_potential, bind_potentials


def generate_solvent_samples(
    coords,
    box,
    masses,
    ubps,
    params,
    temperature,
    pressure,
    seed,
    n_samples,
    num_equil_steps=50000,
    md_steps_per_move=1000,
):
    """TODO: document me"""
    xvb0 = enhanced.equilibrate_solvent_phase(
        ubps, params, masses, coords, box, temperature, pressure, num_equil_steps, seed
    )

    lamb = 1.0  # non-interacting state
    npt_mover = moves.NPTMove(ubps, lamb, masses, temperature, pressure, n_steps=md_steps_per_move, seed=seed)

    xvbs = [xvb0]
    for _ in tqdm(range(n_samples), desc="generating solvent samples"):
        xvbs.append(npt_mover.move(xvbs[-1]))
    return xvbs


def generate_ligand_samples(num_batches, mol, ff, temperature, seed):
    """TODO: document me"""
    state = enhanced.VacuumState(mol, ff)
    proposal_U = state.U_full
    vacuum_samples, vacuum_log_weights = enhanced.generate_log_weighted_samples(
        mol, temperature, state.U_easy, proposal_U, num_batches=num_batches, seed=seed
    )

    return vacuum_samples, vacuum_log_weights


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
