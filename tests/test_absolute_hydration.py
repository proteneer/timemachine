import functools
from typing import Tuple

import numpy as np
import pymbar
import pytest

from timemachine import testsystems
from timemachine.constants import BOLTZ, DEFAULT_FF
from timemachine.fe import absolute_hydration
from timemachine.fe.functional import construct_differentiable_interface_fast
from timemachine.fe.reweighting import one_sided_exp
from timemachine.ff import Forcefield
from timemachine.md import enhanced, moves, smc
from timemachine.md.states import CoordsVelBox


def get_ff_am1ccc():
    return Forcefield.load_from_file(DEFAULT_FF)


@pytest.mark.parametrize("reverse", [False, True])
def test_smc_parameter_change_vacuum(reverse):
    """
    Tests correctness of using SMC to switch between parameters.
    The SMC predicted dg ff0 -> ff1 should match results from
    BAR using independent sampling.

    """
    seed = 2022
    np.random.seed(seed)

    # Reduce these so the test is faster
    num_batches = 10
    n_md_steps = 10
    temperature = 300.0
    kBT = BOLTZ * temperature

    mol, _ = testsystems.ligands.get_biphenyl()
    ff0 = get_ff_am1ccc()

    # shift the ff parameters
    # should be small enough so that 1-EXP is still valid
    ff1 = get_ff_am1ccc()
    ff1.q_handle.params += 0.2
    box = np.eye(3) * 1000.0

    if reverse:
        ff0, ff1 = ff1, ff0

    # generate vacuum samples for each ff
    def generate_samples(mol, ff) -> Tuple[enhanced.VacuumState, smc.Samples]:
        state = enhanced.VacuumState(mol, ff)
        _vacuum_xv_samples, vacuum_log_weights = enhanced.generate_log_weighted_samples(
            mol,
            temperature,
            state.U_easy,
            state.U_full,
            seed,
            num_batches=num_batches,
            num_workers=1,
            steps_per_batch=n_md_steps,
        )

        # discard velocities: (x, v) -> x
        vacuum_samples = _vacuum_xv_samples[:, 0, :]
        vacuum_vels = _vacuum_xv_samples[:, 1, :]
        idxs = smc.multinomial_resample(vacuum_log_weights)[0]
        vacuum_samples = [CoordsVelBox(coords=vacuum_samples[i], velocities=vacuum_vels[i], box=box) for i in idxs]
        return state, vacuum_samples

    state0, vacuum_samples0 = generate_samples(mol, ff0)
    state1, vacuum_samples1 = generate_samples(mol, ff1)

    # set up a system for sampling the new parameters using SMC
    samples, lambdas, propagate, log_prob, resample = absolute_hydration.set_up_ahfe_system_for_smc_parameter_changes(
        mol,
        n_walkers=num_batches,
        n_md_steps=n_md_steps,
        resample_thresh=0.6,
        initial_samples=vacuum_samples0,
        ff0=ff0,
        ff1=ff1,
        is_vacuum=True,
        n_windows=10,
    )

    smc_result = smc.sequential_monte_carlo(samples, lambdas, propagate, log_prob, resample)
    smc_dg = one_sided_exp(-smc_result["log_weights_traj"][-1])

    # BAR from independent runs
    def batched_u_fxn(samples, state):
        # Only compute vacuum energies
        return np.array([state.U_full(x.coords) for x in samples]) / kBT

    u_fwd = batched_u_fxn(vacuum_samples0, state1) - batched_u_fxn(vacuum_samples0, state0)
    u_rev = batched_u_fxn(vacuum_samples1, state0) - batched_u_fxn(vacuum_samples1, state1)
    bar_dg = pymbar.BAR(u_fwd, u_rev, compute_uncertainty=False)
    assert smc_dg == pytest.approx(bar_dg, abs=1e-1)


def test_smc_parameter_change_solvent():
    """
    Tests correctness of using SMC to switch between parameters.
    The SMC predicted dg ff0 -> ff1 should match results from
    BAR using independent sampling.

    """
    seed = 2022
    np.random.seed(seed)

    # Reduce these so the test is faster
    num_batches = 20
    n_md_steps = 100
    temperature = 300.0
    pressure = 1.0
    n_eq_steps = 1000
    kBT = BOLTZ * temperature

    mol, _ = testsystems.ligands.get_biphenyl()
    ff0 = get_ff_am1ccc()

    # shift the ff parameters
    # should be small enough so that 1-EXP is still valid
    ff1 = get_ff_am1ccc()
    ff1.q_handle.params += 0.2

    def generate_samples(mol, ff):  # -> Tuple[Ene_fxn, smc.Samples]:
        ubps, params, masses, coords, box = enhanced.get_solvent_phase_system(mol, ff, minimize_energy=True)
        xvb0 = enhanced.equilibrate_solvent_phase(
            ubps, params, masses, coords, box, temperature, pressure, n_eq_steps, seed
        )
        lam = 0.0
        npt_mover = moves.NPTMove(ubps, lam, masses, temperature, pressure, n_steps=n_md_steps, seed=seed)
        xvbs = [xvb0]
        for _ in range(num_batches):
            xvbs.append(npt_mover.move(xvbs[-1]))

        U_fn = construct_differentiable_interface_fast(ubps, params)
        U_fn = functools.partial(U_fn, params=params)
        return U_fn, xvbs

    U_fn0, solvent_samples0 = generate_samples(mol, ff0)
    U_fn1, solvent_samples1 = generate_samples(mol, ff1)

    # set up a system for sampling the new parameters using SMC
    samples, lambdas, propagate, log_prob, resample = absolute_hydration.set_up_ahfe_system_for_smc_parameter_changes(
        mol,
        n_walkers=num_batches,
        n_md_steps=n_md_steps,
        resample_thresh=0.6,
        initial_samples=solvent_samples0,
        ff0=ff0,
        ff1=ff1,
        is_vacuum=False,
        n_windows=10,
    )

    smc_result = smc.sequential_monte_carlo(samples, lambdas, propagate, log_prob, resample)
    smc_dg = one_sided_exp(-smc_result["log_weights_traj"][-1])

    # BAR from independent runs
    def batched_u_fxn(samples, U_fn):
        # Only compute vacuum energies
        return np.array([U_fn(x.coords, box=x.box, lam=0.0) for x in samples]) / kBT

    u_fwd = batched_u_fxn(solvent_samples0, U_fn1) - batched_u_fxn(solvent_samples0, U_fn0)
    u_rev = batched_u_fxn(solvent_samples1, U_fn0) - batched_u_fxn(solvent_samples1, U_fn1)
    bar_dg = pymbar.BAR(u_fwd, u_rev, compute_uncertainty=False)
    assert smc_dg == pytest.approx(bar_dg, abs=1e-1)
