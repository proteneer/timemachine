from jax.config import config
import pickle
import os
import time

from md.states import CoordsVelBox

config.update("jax_enable_x64", True)

from md import enhanced
from md.moves import NPTMove

import numpy as np
from tests import test_ligands

import jax
from timemachine.potentials import bonded

from fe import functional
from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers

# (ytz): useful for visualization, so please leave this comment here!
import asciiplotlib as apl


def get_ff_am1cc():
    ff_handlers = deserialize_handlers(open("ff/params/smirnoff_1_1_0_ccc.py").read())
    ff = Forcefield(ff_handlers)
    return ff


def generate_solvent_samples(coords, box, masses, ubps, params, temperature, pressure, seed, n_samples):
    num_equil_steps = 5000  # bump to 50k to be safe/production
    xvb0 = enhanced.equilibrate_solvent_phase(
        ubps, params, masses, coords, box, temperature, pressure, num_equil_steps, seed
    )

    md_steps_per_move = 1000  # probably good enough?
    lamb = 1.0  # non-interacting state
    npt_mover = NPTMove(ubps, lamb, masses, temperature, pressure, n_steps=md_steps_per_move, seed=seed)
    xvbs = []
    xvb_t = xvb0
    for _ in range(n_samples):
        xvb_t = npt_mover.move(xvb_t)
        xvbs.append(xvb_t)
    return xvbs


def generate_ligand_samples(num_batches, mol, ff, temperature, seed):
    state = enhanced.VacuumState(mol, ff)
    proposal_U = state.U_full
    vacuum_samples, vacuum_log_weights = enhanced.generate_log_weighted_samples(
        mol, temperature, state.U_easy, proposal_U, num_batches=num_batches, seed=seed
    )

    return vacuum_samples, vacuum_log_weights


def generate_endstate_samples(num_samples, solvent_samples, ligand_samples, ligand_log_weights, num_ligand_atoms):
    all_xbs = []
    for _ in range(num_samples):
        choice_idx = np.random.choice(np.arange(len(solvent_samples)))
        solvent_x = solvent_samples[choice_idx].coords
        ligand_x = enhanced.sample_from_log_weights(ligand_samples, ligand_log_weights, size=1)[0]
        combined_x = np.concatenate([solvent_x[:-num_ligand_atoms], ligand_x], axis=0)
        combined_box = solvent_samples[choice_idx].box
        all_xbs.append((combined_x, combined_box))
    return all_xbs


def test_smc():
    """
    Generate samples from the equilibrium distribution at lambda=1
    """

    seed = 2021
    np.random.seed(seed)

    mol, torsion_idxs = test_ligands.get_biphenyl()

    @jax.jit
    def get_torsion(x_l):
        ci = x_l[torsion_idxs[:, 0]]
        cj = x_l[torsion_idxs[:, 1]]
        ck = x_l[torsion_idxs[:, 2]]
        cl = x_l[torsion_idxs[:, 3]]
        # last [0] is used to return from a length-1 array
        return bonded.signed_torsion_angle(ci, cj, ck, cl)[0]

    ff = get_ff_am1cc()

    ligand_masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    num_ligand_atoms = len(ligand_masses)

    temperature = 300.0
    pressure = 1.0

    ubps, params, masses, coords, box = enhanced.get_solvent_phase_system(mol, ff)

    cache_path = "test_smc_cache.pkl"
    if not os.path.exists(cache_path):
        print("Generating cache")

        n_solvent_samples = 100  # should be maybe 1000
        print(f"generating {n_solvent_samples} solvent samples")
        solvent_xvbs = generate_solvent_samples(
            coords, box, masses, ubps, params, temperature, pressure, seed, n_solvent_samples
        )

        n_ligand_batches = 1200  # should be 30k
        print(f"generating ligand samples")
        ligand_samples, ligand_log_weights = generate_ligand_samples(n_ligand_batches, mol, ff, temperature, seed)

        with open(cache_path, "wb") as fh:
            pickle.dump([solvent_xvbs, ligand_samples, ligand_log_weights], fh)

    with open(cache_path, "rb") as fh:
        print("Loading cache")
        solvent_xvbs, ligand_samples, ligand_log_weights = pickle.load(fh)

    n_endstate_samples = 5000
    all_xbs = generate_endstate_samples(
        n_endstate_samples, solvent_xvbs, ligand_samples, ligand_log_weights, num_ligand_atoms
    )

    # plot torsions at the end-states
    end_state_torsions = []
    for xb in all_xbs:
        x = xb[0]
        x_l = x[:-num_ligand_atoms]
        end_state_torsions.append(get_torsion(x_l))

    # plot histogram using asciiplotlib
    fig = apl.figure()
    fig.hist(*np.histogram(end_state_torsions, bins=25, range=(-np.pi, np.pi)))
    fig.show()

    bound_impls = []
    for u, p in zip(ubps, params):
        bound_impls.append(u.bind(p).bound_impl(np.float32))

    # propagate with NPTMove
    n_steps = 1000
    npt_lamb = 1.0
    seed = int(time.time())

    mover = NPTMove(ubps, npt_lamb, masses, temperature, pressure, n_steps, seed)

    U_fn = functional.construct_differentiable_interface_fast(ubps, params)

    for xi, bi in all_xbs:
        # (ytz): note to jfass, change v0 to not zeros_like
        xvb = CoordsVelBox(xi, np.zeros_like(xi), bi)
        xvb_new = mover.move(xvb)

        # compare delta_Us when coords/box change (this will large and size extensive, probably not useful)
        print(U_fn(xvb_new.coords, params, xvb_new.box, npt_lamb) - U_fn(xvb.coords, params, xvb.box, npt_lamb))

    # or, compare delta_Us when lambda changes

    for xi, bi in all_xbs:
        print(U_fn(xi, params, bi, 0.9) - U_fn(xi, params, bi, 1.0))
