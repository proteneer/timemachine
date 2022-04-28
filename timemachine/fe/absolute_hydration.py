"""Absolute hydration free energies"""

from functools import partial
from typing import List, Sequence

import numpy as np
from numpy.typing import NDArray as Array

from timemachine.constants import BOLTZ
from timemachine.fe import functional
from timemachine.fe.lambda_schedule import construct_pre_optimized_absolute_lambda_schedule_solvent
from timemachine.ff import Forcefield
from timemachine.md import enhanced, moves
from timemachine.md.smc import conditional_multinomial_resample
from timemachine.md.states import CoordsVelBox


def generate_endstate_samples(
    num_samples: int,
    solvent_samples: Sequence[CoordsVelBox],
    ligand_samples: Sequence[CoordsVelBox],
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
    solvent_choice_idxs = np.random.choice(len(solvent_samples), size=num_samples)

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
    ff = Forcefield.load_from_file("smirnoff_1_1_0_ccc.py")
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
    np.random.seed(seed)

    sample_inds = np.random.choice(np.arange(len(initial_samples)), size=n_walkers)
    samples = [initial_samples[i] for i in sample_inds]

    # note: tm convention lambda=1 means "decoupled", lambda=0 means "coupled"
    lambdas = construct_pre_optimized_absolute_lambda_schedule_solvent(n_windows)[::-1]
    assert np.isclose(lambdas[0], 1.0) and np.isclose(lambdas[-1], 0.0)

    def propagate(xs, lam):
        mover.lamb = lam
        xs_next = [mover.move(x) for x in xs]
        return xs_next

    def log_prob(xs, lam):
        u_s = np.array([reduced_potential(x, lam) for x in xs])
        return -u_s

    resample = partial(conditional_multinomial_resample, thresh=resample_thresh)

    return samples, lambdas, propagate, log_prob, resample
