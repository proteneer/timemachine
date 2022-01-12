from jax.config import config

config.update("jax_enable_x64", True)
import jax

from typing import List

from md.states import CoordsVelBox
from md import enhanced
from md.moves import NPTMove, ReferenceMTMMove, OptimizedMTMMove

import functools

from timemachine.potentials import nonbonded
from timemachine.constants import BOLTZ

import numpy as np
from tests import test_ligands
import copy

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers

import jax.numpy as jnp
import jax.random as jrandom


def get_ff_am1ccc():
    ff_handlers = deserialize_handlers(open("ff/params/smirnoff_1_1_0_ccc.py").read())
    ff = Forcefield(ff_handlers)
    return ff


def test_optimized_MTM():
    """
    Tests correctness of an optimized MTM in the condensed phase.
    """

    seed = 2021
    np.random.seed(seed)

    mol, _ = test_ligands.get_biphenyl()
    ff = get_ff_am1ccc()

    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    num_ligand_atoms = len(masses)

    temperature = 300.0
    pressure = 1.0

    state = enhanced.VacuumState(mol, ff)

    proposal_U = state.U_decharged
    num_batches = 480

    vacuum_samples, vacuum_log_weights = enhanced.generate_log_weighted_samples(
        mol, temperature, state.U_easy, proposal_U, seed, num_batches=num_batches
    )

    ubps, params, masses, coords, box = enhanced.get_solvent_phase_system(mol, ff)

    nb_potential = ubps[-1]
    beta = nb_potential.get_beta()
    cutoff = nb_potential.get_cutoff()
    nb_params = params[-1]

    params_i = nb_params[-num_ligand_atoms:]  # ligand params
    params_j = nb_params[:-num_ligand_atoms]  # water params

    kT = temperature * BOLTZ

    def log_Q_a_b_fn(xvb_a, xvb_b):
        x_a = xvb_a.coords
        x_b = xvb_b.coords
        # double check this later
        np.testing.assert_equal(x_a[:-num_ligand_atoms], x_b[:-num_ligand_atoms])
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

    lamb = 0.0
    # we should initialize new instances of this
    npt_mover = NPTMove(ubps, lamb, masses, temperature, pressure, n_steps=1000, seed=seed)

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
