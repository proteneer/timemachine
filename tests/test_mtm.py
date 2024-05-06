import copy
import functools
from typing import List

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from timemachine import testsystems
from timemachine.constants import BOLTZ
from timemachine.fe.utils import get_mol_masses
from timemachine.ff import Forcefield
from timemachine.md import enhanced
from timemachine.md.barostat.moves import NPTMove
from timemachine.md.moves import OptimizedMTMMove, ReferenceMTMMove
from timemachine.md.states import CoordsVelBox
from timemachine.potentials import NonbondedInteractionGroup, nonbonded


def get_ff_am1ccc():
    ff = Forcefield.load_default()
    return ff


@pytest.mark.nightly
def test_optimized_MTM():
    """
    Tests correctness of an optimized MTM in the condensed phase.
    """

    seed = 2021
    np.random.seed(seed)

    mol, _ = testsystems.ligands.get_biphenyl()
    ff = get_ff_am1ccc()

    masses = get_mol_masses(mol)
    num_ligand_atoms = len(masses)

    temperature = 300.0
    pressure = 1.0

    state = enhanced.VacuumState(mol, ff)

    proposal_U = state.U_decharged
    num_batches = 485

    _vacuum_xv_samples, vacuum_log_weights = enhanced.generate_log_weighted_samples(
        mol, temperature, state.U_easy, proposal_U, seed, num_batches=num_batches
    )

    # discard velocities: (x, v) -> x
    vacuum_samples = _vacuum_xv_samples[:, 0, :]
    ubps, params, masses, coords, box = enhanced.get_solvent_phase_system(mol, ff, 0.0)

    # Unwrap SummedPotential to get intermolecular water-ligand potential
    nb_idx, nb_wl_potential = next(
        (i, pot) for i, pot in enumerate(ubps[-1].potentials) if isinstance(pot, NonbondedInteractionGroup)
    )

    nb_params = ubps[-1].params_init[nb_idx]

    beta = nb_wl_potential.beta
    cutoff = nb_wl_potential.cutoff

    params_i = nb_params[-num_ligand_atoms:]  # ligand params
    params_j = nb_params[:-num_ligand_atoms]  # water params

    kT = temperature * BOLTZ

    def log_Q_a_b_fn(xvb_a, xvb_b):
        x_a = xvb_a.coords
        x_b = xvb_b.coords
        # double check this later
        np.testing.assert_array_equal(x_a[:-num_ligand_atoms], x_b[:-num_ligand_atoms])
        return -proposal_U(x_b[-num_ligand_atoms:]) / kT

    def batch_log_Q_a_b_fn(xvbs_a, xvb_b):
        return log_Q_a_b_fn(xvbs_a[0], xvb_b) * np.ones(len(xvbs_a))

    #  should this be proposal_U or full_U?
    def log_lambda_a_b_fn_coords(x_a, x_b):
        x_a_l = x_a[-num_ligand_atoms:]
        x_b_l = x_b[-num_ligand_atoms:]
        return (proposal_U(x_a_l) + proposal_U(x_b_l)) / kT

    batch_log_lambda_a_b_fn_coords = jax.vmap(log_lambda_a_b_fn_coords, in_axes=(0, None))

    def batch_log_lambda_a_b_fn(xvbs_a, xvb_b):
        xs_a = []
        for a in xvbs_a:
            xs_a.append(a.coords)
        xs_a = np.array(xs_a)

        return batch_log_lambda_a_b_fn_coords(xs_a, xvb_b.coords)

    # (ytz): This only works for the decharged case since samples are generated from a decharged state.
    # U_self + U_nonbonded
    def log_prob_fn(x_solvent, box_solvent):
        x_water = x_solvent[:-num_ligand_atoms]  # water coords
        x_ligand = x_solvent[-num_ligand_atoms:]  # ligand coords
        U_wl = nonbonded.nonbonded_block(x_ligand, x_water, box_solvent, params_i, params_j, beta, cutoff)
        U_l = state.U_full(x_ligand)
        return -(U_wl + U_l) / kT

    batch_log_prob_fn = jax.vmap(log_prob_fn)

    # do not take velocities into account when evaluating log probabilities
    def batch_log_prob_wrapper(xvbs: List[CoordsVelBox]) -> List[float]:
        batch_coords = np.array([xvb.coords for xvb in xvbs])
        batch_boxes = np.array([xvb.box for xvb in xvbs])
        return batch_log_prob_fn(batch_coords, batch_boxes)

    # for use in optimized method
    def log_weights_fn(x, box):
        x_water = x[:-num_ligand_atoms]  # water coords
        x_ligand = x[-num_ligand_atoms:]  # ligand coords
        U_wl = nonbonded.nonbonded_block(x_ligand, x_water, box, params_i, params_j, beta, cutoff)
        U_l = state.U_full(x_ligand)
        U_p = proposal_U(x_ligand)
        return -(U_wl + U_l - U_p) / kT

    batch_log_weights_fn = jax.jit(jax.vmap(log_weights_fn, in_axes=(0, None)))

    num_equil_steps = 5000
    xvb0 = enhanced.equilibrate_solvent_phase(
        ubps, params, masses, coords, box, temperature, pressure, num_equil_steps, seed
    )
    batch_proposal_fn = functools.partial(
        enhanced.aligned_batch_propose, vacuum_samples=vacuum_samples, vacuum_log_weights=vacuum_log_weights
    )
    batch_proposal_coords_fn = functools.partial(
        enhanced.jax_aligned_batch_propose_coords,
        vacuum_samples=jnp.array(vacuum_samples),
        vacuum_log_weights=jnp.array(vacuum_log_weights),
    )

    bps = [ubp.bind(params) for ubp, params in zip(ubps, params)]

    # we should initialize new instances of this
    npt_mover = NPTMove(bps, masses, temperature, pressure, n_steps=1000, seed=seed)

    K = 100
    # note that these seeds aren't actually used, since we feed in explicit keys to acceptance_probability
    ref_mtm_mover = ReferenceMTMMove(
        K, batch_proposal_fn, batch_log_Q_a_b_fn, batch_log_prob_wrapper, batch_log_lambda_a_b_fn, seed=seed
    )
    test_mtm_mover = OptimizedMTMMove(K, batch_proposal_coords_fn, batch_log_weights_fn, seed=seed)
    xvb_t = copy.deepcopy(xvb0)
    key = jrandom.PRNGKey(0)

    num_batches = 15
    for _ in range(num_batches):
        xvb_t = npt_mover.move(xvb_t)

        ref_yvb, ref_prob, ref_key = ref_mtm_mover.acceptance_probability(xvb_t, key)

        np.testing.assert_array_equal(ref_yvb.velocities, xvb_t.velocities)
        np.testing.assert_array_equal(ref_yvb.box, xvb_t.box)

        test_y, test_prob, test_key = test_mtm_mover.acceptance_probability(xvb_t.coords, xvb_t.box, key)

        np.testing.assert_array_equal(ref_yvb.coords, test_y)
        np.testing.assert_almost_equal(ref_prob, test_prob)
        np.testing.assert_array_equal(ref_key, test_key)

        _, key = jrandom.split(ref_key)
