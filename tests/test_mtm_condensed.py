import copy
import functools
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from timemachine import testsystems
from timemachine.constants import BOLTZ, DEFAULT_PRESSURE, DEFAULT_TEMP
from timemachine.fe.utils import get_mol_masses
from timemachine.ff import Forcefield
from timemachine.md import enhanced
from timemachine.md.barostat.moves import NPTMove
from timemachine.md.moves import NVTMove, OptimizedMTMMove
from timemachine.md.states import CoordsVelBox
from timemachine.potentials import NonbondedInteractionGroup, SummedPotential, bonded, nonbonded
from timemachine.potentials.potential import get_potential_by_type


@pytest.mark.skip("Has shown to be flaky, needs further investigation. Condensed MTM not currently used")
@pytest.mark.nightly(reason="Takes a long time to run")
@pytest.mark.parametrize("seed", [2021])
def test_condensed_phase_mtm(seed):
    """
    Tests multiple-try metropolis in the condensed phase.
    """

    np.random.seed(seed)

    mol, torsion_idxs = testsystems.ligands.get_biphenyl()
    ff = Forcefield.load_default()

    masses = get_mol_masses(mol)
    num_ligand_atoms = len(masses)

    temperature = DEFAULT_TEMP
    pressure = DEFAULT_PRESSURE

    state = enhanced.VacuumState(mol, ff)

    proposal_U = state.U_decharged

    num_workers = 4
    weighted_batches = 400000
    cache_path = f"mtm_condensed_cache_seed_{seed}_workers_{num_workers}_batches_{weighted_batches}.pkl"
    if not os.path.exists(cache_path):
        print("Generating cache")
        vacuum_samples, vacuum_log_weights = enhanced.generate_log_weighted_samples(
            mol, temperature, state.U_easy, proposal_U, num_batches=weighted_batches, seed=seed, num_workers=num_workers
        )
        # Discard velocities
        vacuum_samples = vacuum_samples[:, 0, :]

        with open(cache_path, "wb") as fh:
            pickle.dump([vacuum_samples, vacuum_log_weights], fh)

    with open(cache_path, "rb") as fh:
        vacuum_samples, vacuum_log_weights = pickle.load(fh)

    ubps, params, masses, coords, box = enhanced.get_solvent_phase_system(mol, ff, 0.0)

    # Unwrap SummedPotential to get water-ligand nonbonded potential
    summed_pot = get_potential_by_type(ubps, SummedPotential)

    # Unwrap SummedPotential to get intermolecular water-ligand potential
    nb_idx, nb_wl_potential = next(
        (i, pot) for i, pot in enumerate(summed_pot.potentials) if isinstance(pot, NonbondedInteractionGroup)
    )

    nb_params = summed_pot.params_init[nb_idx]

    beta = nb_wl_potential.beta
    cutoff = nb_wl_potential.cutoff

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

    # import matplotlib.pyplot as plt

    # plt.hist([vacuum_torsions], bins=50, density=True, weights=[vacuum_weights])
    # plt.show()

    assert np.abs(np.average(vacuum_torsions, weights=vacuum_weights)) < 0.15

    num_batches = 3333
    md_steps_per_move = 150

    print(f"Running a total of {num_batches * md_steps_per_move} md steps")

    all_torsions = []

    num_equil_steps = 30000
    xvb0 = enhanced.equilibrate_solvent_phase(
        ubps, params, masses, coords, box, temperature, pressure, num_equil_steps, seed
    )

    # (ytz): emprically scanning over multiple Ks seem to suggest 100 is a sweet spot
    # leave this here for pedagogical purposes.
    # for K in [1, 5, 10, 25, 50, 100, 200, 400]:

    K = 100

    batch_proposal_coords_fn = functools.partial(
        enhanced.jax_aligned_batch_propose_coords,
        vacuum_samples=jnp.array(vacuum_samples),
        vacuum_log_weights=jnp.array(vacuum_log_weights),
    )

    bps = [ubp.bind(params) for ubp, params in zip(ubps, params)]

    npt_mover = NPTMove(bps, masses, temperature, pressure, n_steps=md_steps_per_move, seed=seed)
    mtm_mover = OptimizedMTMMove(K, batch_proposal_coords_fn, batch_log_weights_fn, seed=seed)

    enhanced_torsions = []

    xvb_t = copy.deepcopy(xvb0)

    for iteration in range(num_batches):
        xvb_t = npt_mover.move(xvb_t)
        solvent_torsion = get_torsion(xvb_t.coords[-num_ligand_atoms:])
        enhanced_torsions.append(solvent_torsion)
        xvb_t = mtm_mover.move(xvb_t)

        print(
            f"K {K} frame {iteration} acceptance rate {mtm_mover.n_accepted / mtm_mover.n_proposed} solvent_torsion {solvent_torsion}"
        )

    all_torsions.append(np.asarray(enhanced_torsions))

    # lhs is (-np.pi, 0) and rhs is (0, np.pi)
    enhanced_torsions_lhs, _ = np.histogram(enhanced_torsions, bins=50, range=(-np.pi, 0), density=True)
    enhanced_torsions_rhs, _ = np.histogram(enhanced_torsions, bins=50, range=(0, np.pi), density=True)

    # check for symmetry about theta=0
    assert np.mean((enhanced_torsions_lhs - enhanced_torsions_rhs[::-1]) ** 2) < 5e-2

    vanilla_torsions = []
    xvb_t = copy.deepcopy(xvb0)
    npt_mover = NPTMove(bps, masses, temperature, pressure, n_steps=500, seed=seed)
    for iteration in range(num_batches):
        solvent_torsion = get_torsion(xvb_t.coords[-num_ligand_atoms:])
        vanilla_torsions.append(solvent_torsion)
        xvb_t = npt_mover.move(xvb_t)

    vanilla_torsions = np.asarray(vanilla_torsions)

    vanilla_samples_rhs, _ = np.histogram(vanilla_torsions, bins=50, range=(0, np.pi), density=True)

    # check for consistency with vanilla samples
    assert np.mean((enhanced_torsions_rhs - vanilla_samples_rhs) ** 2) < 5e-2


def test_nvt_box():
    # Test box stays the same under NVTMove
    seed = 2022
    np.random.seed(seed)

    mol, _ = testsystems.ligands.get_biphenyl()
    ff = Forcefield.load_default()

    ubps, params, masses, coords, box = enhanced.get_solvent_phase_system(mol, ff, 0.0, minimize_energy=False)
    bps = []
    for p, bp in zip(params, ubps):
        bps.append(bp.bind(p))

    temperature = DEFAULT_TEMP
    n_steps = 100
    mover = NVTMove(bps, masses, temperature, n_steps, seed)
    v0 = np.zeros_like(coords)
    xvb0 = CoordsVelBox(coords, v0, box)
    xvb = mover.move(xvb0)
    assert np.allclose(box, xvb.box)
