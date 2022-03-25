import os
import pickle

from jax.config import config
from scipy.special import logsumexp

config.update("jax_enable_x64", True)
import copy
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_ligands

from timemachine.constants import BOLTZ
from timemachine.ff import Forcefield
from timemachine.md import enhanced
from timemachine.md.moves import NPTMove, OptimizedMTMMove
from timemachine.potentials import bonded, nonbonded

# (ytz): useful for visualization, so please leave this comment here!


def get_ff_am1cc():
    ff = Forcefield.load_from_file("smirnoff_1_1_0_ccc.py")
    return ff


@pytest.mark.skip(reason="This takes too long to run on CI")
def test_condensed_phase_mtm():
    """
    Tests multiple-try metropolis in the condensed phase.
    """

    seed = 2021
    np.random.seed(seed)

    mol, torsion_idxs = test_ligands.get_biphenyl()
    ff = get_ff_am1cc()

    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    num_ligand_atoms = len(masses)

    temperature = 300.0
    pressure = 1.0

    state = enhanced.VacuumState(mol, ff)

    proposal_U = state.U_decharged
    seed = 2021

    cache_path = "cache.pkl"
    if not os.path.exists(cache_path):
        print("Generating cache")
        num_batches = 30000
        vacuum_samples, vacuum_log_weights = enhanced.generate_log_weighted_samples(
            mol, temperature, state.U_easy, proposal_U, num_batches=num_batches, seed=seed
        )

        with open(cache_path, "wb") as fh:
            pickle.dump([vacuum_samples, vacuum_log_weights], fh)

    with open(cache_path, "rb") as fh:
        vacuum_samples, vacuum_log_weights = pickle.load(fh)

    ubps, params, masses, coords, box = enhanced.get_solvent_phase_system(mol, ff)

    nb_potential = ubps[-1]
    beta = nb_potential.get_beta()
    cutoff = nb_potential.get_cutoff()
    nb_params = params[-1]

    params_i = nb_params[-num_ligand_atoms:]  # ligand params
    params_j = nb_params[:-num_ligand_atoms]  # water params

    # sanity check that charges on ligand is zero
    # assert np.all(params_i[:, 0] == 0.0)

    kT = temperature * BOLTZ

    # for use in optimized method
    def log_weights_fn(x, box):
        x_water = x[:-num_ligand_atoms]  # water coords
        x_ligand = x[-num_ligand_atoms:]  # ligand coords
        U_wl = nonbonded.nonbonded_block(x_ligand, x_water, box, params_i, params_j, beta, cutoff)
        U_l = state.U_full(x_ligand)
        U_p = proposal_U(x_ligand)
        return -(U_wl + U_l - U_p) / kT

    batch_log_weights_fn = jax.jit(jax.vmap(log_weights_fn, in_axes=(0, None)))

    @jax.jit
    def get_torsion(x_l):
        ci = x_l[torsion_idxs[:, 0]]
        cj = x_l[torsion_idxs[:, 1]]
        ck = x_l[torsion_idxs[:, 2]]
        cl = x_l[torsion_idxs[:, 3]]
        return bonded.signed_torsion_angle(ci, cj, ck, cl)

    batch_get_torsion = jax.jit(jax.vmap(get_torsion))
    vacuum_weights = np.exp(vacuum_log_weights - logsumexp(vacuum_log_weights))
    assert np.abs(np.sum(vacuum_weights) - 1) < 1e-5
    vacuum_torsions = batch_get_torsion(vacuum_samples).reshape(-1)

    import matplotlib.pyplot as plt

    plt.hist(vacuum_torsions, bins=50, density=True, weights=vacuum_weights)
    plt.show()

    assert np.abs(np.average(vacuum_torsions, weights=vacuum_weights)) < 0.1

    num_batches = 3333
    md_steps_per_move = 150

    print(f"Running a total of {num_batches*md_steps_per_move} md steps")

    frozen_masses = np.copy(masses)
    frozen_masses[-num_ligand_atoms:] = np.inf

    all_torsions = []

    num_equil_steps = 30000
    xvb0 = enhanced.equilibrate_solvent_phase(
        ubps, params, masses, coords, box, temperature, pressure, num_equil_steps, seed
    )

    # test with both frozen masses and free masses
    for test_masses in [frozen_masses, masses]:

        # (ytz): emprically scanning over multiple Ks seem to suggest 50 is a sweet spot
        # leave this here for pedagogical purposes.
        # for K in [1, 5, 10, 25, 50, 100, 200, 400]:

        K = 100

        batch_proposal_coords_fn = functools.partial(
            enhanced.jax_aligned_batch_propose_coords,
            vacuum_samples=jnp.array(vacuum_samples),
            vacuum_log_weights=jnp.array(vacuum_log_weights),
        )

        lamb = 0.0

        npt_mover = NPTMove(ubps, lamb, test_masses, temperature, pressure, n_steps=md_steps_per_move, seed=seed)
        mtm_mover = OptimizedMTMMove(K, batch_proposal_coords_fn, batch_log_weights_fn, seed=seed)

        enhanced_torsions = []

        xvb_t = copy.deepcopy(xvb0)

        for iteration in range(num_batches):

            xvb_t = npt_mover.move(xvb_t)
            solvent_torsion = get_torsion(xvb_t.coords[-num_ligand_atoms:])
            enhanced_torsions.append(solvent_torsion)
            xvb_t = mtm_mover.move(xvb_t)

            print(
                f"K {K} frame {iteration} acceptance rate {mtm_mover.n_accepted/mtm_mover.n_proposed} solvent_torsion {solvent_torsion}"
            )

        all_torsions.append(np.asarray(enhanced_torsions))

    # for biphenyl, reshape into -1
    all_torsions = np.asarray(all_torsions).reshape(-1)

    enhanced_torsions_lhs, _ = np.histogram(enhanced_torsions, bins=50, range=(-np.pi, 0), density=True)
    enhanced_torsions_rhs, _ = np.histogram(enhanced_torsions, bins=50, range=(0, np.pi), density=True)

    # check for symmetry about theta=0
    assert np.mean((enhanced_torsions_lhs - enhanced_torsions_rhs[::-1]) ** 2) < 5e-2

    vanilla_torsions = []
    xvb_t = copy.deepcopy(xvb0)
    npt_mover = NPTMove(ubps, lamb, test_masses, temperature, pressure, n_steps=500, seed=seed)
    for iteration in range(num_batches):
        solvent_torsion = get_torsion(xvb_t.coords[-num_ligand_atoms:])
        vanilla_torsions.append(solvent_torsion)
        xvb_t = npt_mover.move(xvb_t)

    vanilla_torsions = np.asarray(vanilla_torsions)

    plt.hist(all_torsions[0], bins=np.linspace(-np.pi, np.pi, 100), density=True, alpha=0.5, label="enhanced_frozen")
    plt.hist(all_torsions[1], bins=np.linspace(-np.pi, np.pi, 100), density=True, alpha=0.5, label="enhanced_free")
    plt.hist(vanilla_torsions, bins=np.linspace(-np.pi, np.pi, 100), density=True, alpha=0.25, label="vanilla")
    plt.legend()
    plt.xlabel("torsion angle")
    plt.ylabel("density")
    plt.show()

    vanilla_samples_lhs, _ = np.histogram(vanilla_torsions, bins=50, range=(-np.pi, 0), density=True)

    # check for consistency with vanilla samples
    assert np.mean((enhanced_torsions_lhs - vanilla_samples_lhs) ** 2) < 5e-2
