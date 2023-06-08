from jax.config import config
from scipy.stats import special_ortho_group

config.update("jax_enable_x64", True)

import timemachine

print(timemachine.__version__)

import argparse
import functools
from copy import deepcopy
from time import time

import numpy as np
from scipy.special import logsumexp

from timemachine import lib
from timemachine.constants import BOLTZ, DEFAULT_TEMP
from timemachine.fe import cif_writer
from timemachine.fe.model_utils import apply_hmr, image_frame
from timemachine.ff import Forcefield
from timemachine.lib import custom_ops
from timemachine.md import enhanced
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.states import CoordsVelBox

temperature = DEFAULT_TEMP
kBT = BOLTZ * temperature


# TODO: load waterbox samples from disk, rather than generating from scratch

from dataclasses import dataclass

from rdkit import Chem
from tqdm import tqdm  # not great with running in parallel...


def run_md(ctxt, n_samples=500, thinning=1000):
    samples = []
    for _ in tqdm(range(n_samples)):
        # for _ in range(n_samples):
        ctxt.multiple_steps(thinning)

        x = ctxt.get_x_t()
        v = ctxt.get_v_t()
        box = ctxt.get_box()
        xvb = CoordsVelBox(x, v, box)
        samples.append(xvb)

    return samples


class ComplexPhaseSystem:
    def __init__(self, mol, host_pdb, ff):
        self.mol = mol
        self.ff = ff
        self.host_pdb = host_pdb

        (
            self.potentials,
            self.params,
            self.masses,
            self.coords,
            self.box,
            self.complex_topology,
            self.ligand_idxs,
            self.ligand_ranks,
            self.jordan_idxs,
            self.rest_idxs,
            self.rest_ranks,
        ) = enhanced.get_complex_phase_system(mol, host_pdb, ff, minimize_energy=True)

    def construct_context(self, params, seed=2022):
        dt = 2.5e-3
        friction = 1.0
        pressure = 1.0
        temperature = DEFAULT_TEMP

        # TODO: speed this up...
        # also, dt = 2.5e-3, with HMR

        bps = []
        for param, potential in zip(params, self.potentials):
            bps.append(potential.bind(param))

        all_impls = [bp.to_gpu(np.float32).bound_impl for bp in bps]

        bond_list = get_bond_list(self.potentials[0])

        masses = apply_hmr(self.masses, bond_list)

        intg = lib.LangevinIntegrator(temperature, dt, friction, masses, seed)
        intg_impl = intg.impl()

        group_idxs = get_group_indices(bond_list, len(masses))
        barostat_interval = 5

        barostat = lib.MonteCarloBarostat(len(masses), pressure, temperature, group_idxs, barostat_interval, seed + 1)
        barostat_impl = barostat.impl(all_impls)

        ctxt = custom_ops.Context(
            self.coords, np.zeros_like(self.coords), self.box, intg_impl, all_impls, barostat_impl
        )

        ctxt.setup_local_md(temperature, False)

        # ugg, otherwise code that returns only `ctxt` will segfault...
        ctxt._all_impls = all_impls
        ctxt._barostat = barostat
        ctxt._barostat_impl = barostat_impl
        ctxt._intg = intg
        ctxt._intg_impl = intg_impl

        return ctxt

    def generate_initial_samples(self, params, n_samples=500, seed=2022, burn_in=100_000, n_eq_steps=10000):
        """bind self.potentials to params, construct contexts, run simulation"""

        print(f"generating samples (seed={seed})")
        ctxt = self.construct_context(params, seed)

        # burn-in
        print(f"\tburning in end-state for {burn_in} steps...")
        t0 = time()

        coords, boxes = ctxt.multiple_steps(burn_in, 0)

        samples = []

        print(f"\tgenerating {n_samples} frames once every {n_eq_steps} steps...")
        for _ in range(n_samples):
            ctxt.multiple_steps(n_eq_steps, 0)

            samples.append(
                CoordsVelBox(
                    ctxt.get_x_t(),
                    ctxt.get_v_t(),
                    ctxt.get_box(),
                )
            )

        return samples

        assert 0

        # _ = ctxt.multiple_steps(burn_in)
        t1 = time()
        print(f"\tdone! ({(t1 - t0):.1f} seconds)")

        # production
        print(f"\tproduction (n_samples={n_samples})")  # , thinning={thinning})...")
        prod_start_time = time()
        samples = run_md(ctxt, n_samples, thinning=1000)
        # samples = run_local_md(ctxt, self.ligand_idxs, n_samples, local_md_config)
        prod_end_time = time()
        prod_elapsed_time = prod_end_time - prod_start_time
        print(f"\tdone! ({prod_elapsed_time:.1f} seconds)")

        return samples


def xvbs_to_arrays(xvbs):
    traj = np.array([xvb.coords for xvb in xvbs])
    boxes = np.array([xvb.box for xvb in xvbs])

    return traj, boxes


# def apply_lifting_old(host_system, lam=0.0):
#     params = deepcopy(host_system.params)
#     nb_params = params[-1]

#     cutoff = 1.2
#     switch = 0.5
#     min_epsilon = 0.02

#     if lam < switch:
#         ligand_w_coords = 0
#         ligand_charges = -2 * nb_params[host_system.ligand_idxs, 0] * (lam - switch)
#         ligand_epsilons = np.clip(-2 * nb_params[host_system.ligand_idxs, 2] * (lam - switch), min_epsilon, np.inf)

#         rest_w_coords = 0
#         rest_charges = -2 * nb_params[host_system.rest_idxs, 0] * (lam - switch)
#         rest_epsilons = np.clip(-2 * nb_params[host_system.rest_idxs, 2] * (lam - switch), min_epsilon, np.inf)
#     else:
#         ligand_w_coords = 2 * (lam - switch) * cutoff  # LIGAND atoms approach from positive w
#         ligand_charges = 0.0
#         ligand_epsilons = min_epsilon

#         rest_w_coords = -2 * (lam - switch) * cutoff  # REST side chains approach from negative w
#         rest_charges = 0.0
#         rest_epsilons = min_epsilon

#     nb_params = nb_params.at[host_system.ligand_idxs, -1].set(ligand_w_coords)
#     nb_params = nb_params.at[host_system.ligand_idxs, 0].set(ligand_charges)
#     nb_params = nb_params.at[host_system.ligand_idxs, 2].set(ligand_epsilons)

#     nb_params = nb_params.at[host_system.rest_idxs, -1].set(rest_w_coords)
#     nb_params = nb_params.at[host_system.rest_idxs, 0].set(rest_charges)
#     nb_params = nb_params.at[host_system.rest_idxs, 2].set(rest_epsilons)

#     # TBD: turn off centroid restraint

#     return tuple(params[:-1]) + (nb_params,)


def _get_w_coords_impl(lam, ranks, cutoff):
    num_atoms = len(ranks)
    bin_width = 1 / num_atoms
    w_coords = np.zeros(num_atoms)

    for atom_idx in range(num_atoms):
        fractional_rank = ranks[atom_idx] / num_atoms
        rhs_bound = 1 - fractional_rank
        lhs_bound = (1 - fractional_rank) - bin_width
        if lam >= rhs_bound:
            w_frac = 0
        elif lam < rhs_bound and lam >= lhs_bound:
            w_frac = (rhs_bound - lam) * num_atoms
        elif lam < lhs_bound:
            w_frac = 1
        else:
            assert 0

        w_coords[atom_idx] = cutoff - w_frac * cutoff

    return w_coords


def get_w_coords(lam, ranks, cutoff):
    if lam < 0.5:
        lam_eff = 0.0
    elif lam >= 0.5:
        lam_eff = lam * 2 - 1
    return _get_w_coords_impl(lam_eff, ranks, cutoff)


# apply insertion sequentially
def apply_lifting(host_system, lam=0.0):
    params = deepcopy(host_system.params)
    nb_params = params[-1]

    cutoff = 1.2
    switch = 0.5
    # min_epsilon = 0.02

    ligand_ranks = host_system.ligand_ranks
    # rest_ranks = host_system.rest_ranks

    ligand_w_coords = get_w_coords(lam, ligand_ranks, cutoff)
    # rest_w_coords = -get_w_coords(lam, rest_ranks, cutoff)

    # print("lam", lam, "ligand_w", ligand_w_coords)
    # print("lam", lam, "rest_w", rest_w_coords)

    # ligand atoms are grown from "inside-out"
    # rest atoms are grown from "outside-in"
    # hopefully this will "smoothly" push out the ligand
    # tbd: shrink sigma too?

    if lam < switch:
        ligand_charges = -2 * nb_params[host_system.ligand_idxs, 0] * (lam - switch)
        # ligand_epsilons = np.clip(-2 * nb_params[host_system.ligand_idxs, 2] * (lam - switch), min_epsilon, np.inf)

        # rest_charges = -2 * nb_params[host_system.rest_idxs, 0] * (lam - switch)
        # rest_epsilons = np.clip(-2 * nb_params[host_system.rest_idxs, 2] * (lam - switch), min_epsilon, np.inf)
    else:
        ligand_charges = 0.0
        # ligand_epsilons = min_epsilon

        # rest_charges = 0.0
        # rest_epsilons = min_epsilon

    nb_params = nb_params.at[host_system.ligand_idxs, -1].set(ligand_w_coords)
    nb_params = nb_params.at[host_system.ligand_idxs, 0].set(ligand_charges)
    # nb_params = nb_params.at[host_system.ligand_idxs, 2].set(ligand_epsilons)

    # nb_params = nb_params.at[host_system.rest_idxs, -1].set(rest_w_coords)
    # nb_params = nb_params.at[host_system.rest_idxs, 0].set(rest_charges)
    # nb_params = nb_params.at[host_system.rest_idxs, 2].set(rest_epsilons)

    # TBD: turn off centroid restraint

    return tuple(params[:-1]) + (nb_params,)


@dataclass
class LocalMDConfig:
    n_steps: int
    local_idxs: np.array
    burn_in: int
    store_x_interval: int
    radius: float
    k: float


# TODO: do all subsequent steps of local propagation with a fixed selection mask?
def local_propagation(xvbs, ctxt, local_md_config):
    updates = []

    for xvb in xvbs:
        seed = np.random.randint(10000)

        # def _local_propagate(xvb, ctxt, local_md_config, seed):
        ctxt.set_box(xvb.box)
        ctxt.set_v_t(xvb.velocities)
        ctxt.set_x_t(xvb.coords)

        ctxt.multiple_steps_local(
            local_md_config.n_steps,
            local_md_config.local_idxs,
            local_md_config.burn_in,
            local_md_config.store_x_interval,
            local_md_config.radius,
            local_md_config.k,
            seed,
        )

        x = ctxt.get_x_t()
        v = ctxt.get_v_t()
        box = ctxt.get_box()

        updates.append(CoordsVelBox(x, v, box))

    return updates


# now, convert from kBT to whatever unit dG is in -- kJ/mol?
from timemachine.fe.reweighting import one_sided_exp
from timemachine.md.smc import conditional_multinomial_resample, effective_sample_size

# how to do summed potential again?
from timemachine.potentials import SummedPotential


def resample_fxn(log_weights, thresh):
    ess = effective_sample_size(log_weights)
    msg = f"""
        ESS = {ess:.3f} ({100 * ess / len(log_weights):.5f}%)
        running estimate = {one_sided_exp(-log_weights):.5f}
    """
    print(msg)
    out = conditional_multinomial_resample(log_weights, thresh)
    # out = conditional_multinomial_resample(log_weights, thresh=0.2)

    return out


from typing import Callable

from timemachine.md.smc import BatchLogProb, BatchPropagator, Lambda, LogWeights, Resampler, ResultDict, Samples

NextLamSelector = Callable[[Samples, LogWeights, Lambda], Lambda]


def adaptive_neq_switch(
    samples: Samples,
    propagate: BatchPropagator,
    log_prob: BatchLogProb,
    resample: Resampler,
    log_weights,
    initial_lam: Lambda = 1.0,
    final_lam: Lambda = 0.0,
    max_num_lambdas: int = 10000,
    callback_fn=None,  # what to do
    callback_interval=10,
) -> ResultDict:
    """barebones implementation of Sequential Monte Carlo (SMC)

    Parameters
    ----------
    samples: [N,] list
    propagate: function
        [move(x, lam) for x in xs]
        for example, move(x, lam) might mean "run 100 steps of all-atom MD targeting exp(-u(., lam)), initialized at x"
    log_prob: function
        [exp(-u(x, lam)) for x in xs]
    resample: function
        (optionally) perform resampling given an array of log weights
    log_weights: [N,]
        list of initial weights

    Returns
    -------
    trajs_dict
        "sample_traj"
            [K-1, N] list of snapshots
        "incremental_log_weights_traj"
            [K-1, N] array of incremental log weights
        "ancestry_traj"
            [K-1, N] array of ancestor idxs
        "log_weights_traj"
            [K, N] array of accumulated log weights

    References
    ----------
    * Arnaud Doucet's annotated bibliography of SMC
        https://www.stats.ox.ac.uk/~doucet/smc_resources.html
    * [Dai, Heng, Jacob, Whiteley, 2020] An invitation to sequential Monte Carlo samplers
        https://arxiv.org/abs/2007.11936

    See Also
    --------
    * get_endstate_samples_from_smc_result
    """

    cur_samples = samples
    log_weights_traj = [log_weights]
    lambdas = [initial_lam]

    for t in range(max_num_lambdas):
        if t % callback_interval == 0:
            callback_fn(t, lambdas[-1], log_weights, cur_samples, final=False)

        if lambdas[-1] == final_lam:
            break

        lam_initial = lambdas[-1]

        # np.zeros_like(log_weights) discards history - bad for engagement rings when "coupling"
        lam_target = select_next_lam_simple(cur_samples, lam_initial, final_lam, batch_log_prob=log_prob)
        # if final_lam == 0.0:
        #     direction = "fwd"
        # elif final_lam == 1.0:
        #     direction = "rev"
        # else:
        #     assert 0
        # lam_target = select_next_lam_NEQ(t + 1, direction=direction)

        # update log weights
        incremental_log_weights = log_prob(cur_samples, lam_target) - log_prob(cur_samples, lam_initial)
        # say no to +=
        log_weights = log_weights + incremental_log_weights

        # resample, disabled for now
        # indices, log_weights = resample(log_weights)
        # resampled = [cur_samples[i] for i in indices]
        # cur_samples = propagate(resampled, lam_target)

        # propagate (also removes some of the clashiness induced by moving to lam_target)
        cur_samples = propagate(cur_samples, lam_target)

        # append
        lambdas.append(lam_target)

        # the copy() probably isn't necessary now that we've removed the +=
        log_weights_traj.append(log_weights.copy())

        if lam_target == final_lam:
            print("Terminating")
            break

    callback_fn(t, lambdas[-1], log_weights, cur_samples, final=True)

    return cur_samples, lambdas, log_weights_traj


from scipy.optimize import root_scalar

# def select_next_lam_CESS(
#     samples,
#     log_weights,
#     current_lam,
#     target_lam,
#     batch_log_prob,
#     frac_ess_reduction_threshold=0.01,
#     xtol=1e-5,
#     verbose=False,
# ):
#     # CESS
#     # implementation adapted from https://github.com/proteneer/timemachine/pull/442

#     # TODO: phrase thresh multiplicatively instead of additively
#     # (ESS_next = beta * ESS, rather than
#     #  ESS_next = ESS - thresh)
#     # as in "SMC with transformations" and prior work

#     ess = effective_sample_size(log_weights)
#     frac_ess = ess / len(log_weights)

#     assert frac_ess >= frac_ess_reduction_threshold

#     log_p_0 = batch_log_prob(samples, current_lam)

#     # note scaled:
#     # current_lam + 1.0 * remainder = target_lam
#     direction = target_lam - current_lam

#     def fractional_ess_as_fxn_of_increment(increment: float) -> float:
#         trial_lam = current_lam + direction * increment
#         log_p_trial = batch_log_prob(samples, trial_lam)
#         incremental_log_weight = log_p_trial - log_p_0
#         trial_log_weights = log_weights + incremental_log_weight
#         trial_ess = effective_sample_size(trial_log_weights) / len(trial_log_weights)
#         if verbose:
#             print(increment, trial_ess)
#         return np.nan_to_num(trial_ess, nan=0.0)

#     def f(lam_increment: float) -> float:
#         frec_ess_reduction = frac_ess - fractional_ess_as_fxn_of_increment(lam_increment)

#         return frec_ess_reduction - frac_ess_reduction_threshold

#     # try-except to catch rootfinding ValueError: f(a) and f(b) must have different signs
#     #   which occurs when jumping all the way to lam=1.0 is still less than threshold
#     try:
#         result = root_scalar(f, bracket=(0, 1.0), xtol=xtol, maxiter=20)
#         lam_increment = result.root
#     except ValueError as e:
#         print(f"root finding error: {e}, terminating and skipping to the end")
#         lam_increment = 1.0

#     next_lam = current_lam + direction * lam_increment

#     return next_lam


def select_next_lam_simple(
    samples,
    current_lam,
    target_lam,
    batch_log_prob,
    frac_ess_reduction_threshold=0.05,
    xtol=1e-5,
    verbose=False,
):
    log_p_0 = batch_log_prob(samples, current_lam)

    # note scaled:
    # current_lam + 1.0 * remainder = target_lam
    direction = target_lam - current_lam

    def fractional_ess_as_fxn_of_increment(increment: float) -> float:
        trial_lam = current_lam + direction * increment
        log_p_trial = batch_log_prob(samples, trial_lam)
        incremental_log_weight = log_p_trial - log_p_0
        # print(
        #     "incremental_log_weights",
        #     incremental_log_weight,
        #     "cur lam",
        #     current_lam,
        #     "trial_lam",
        #     trial_lam,
        #     "with increment",
        #     increment,
        # )
        trial_log_weights = incremental_log_weight
        trial_ess = effective_sample_size(trial_log_weights) / len(trial_log_weights)
        if verbose:
            print(increment, trial_ess)
        return np.nan_to_num(trial_ess, nan=0.0)

    def f(lam_increment: float) -> float:
        frec_ess_reduction = 1.0 - fractional_ess_as_fxn_of_increment(lam_increment)
        return frec_ess_reduction - frac_ess_reduction_threshold

    # print("start root_scalar", current_lam)
    # try-except to catch rootfinding ValueError: f(a) and f(b) must have different signs
    #   which occurs when jumping all the way to lam=1.0 is still less than threshold
    try:
        result = root_scalar(f, bracket=(0, 1.0), xtol=xtol, maxiter=20)
        lam_increment = result.root
    except ValueError as e:
        print(f"root finding error: {e}, terminating and skipping to the end")
        lam_increment = 1.0

    next_lam = current_lam + direction * lam_increment

    return next_lam


def select_next_lam_NEQ(idx, direction):
    precomputed = []  # empty for now
    if direction == "fwd":
        return precomputed[idx]
    elif direction == "rev":
        return precomputed[::-1][idx]
    else:
        assert 0


def make_smc_funcs(system, local_md_config):
    U_fn = SummedPotential(system.potentials, system.params).to_gpu(np.float32).call_with_params_list

    def u(x, box, params):
        return U_fn(x, params, box) / kBT

    def batch_u(samples, params):
        return np.array([u(s.coords, s.box, params) for s in samples])

    def batch_propagate(samples, lam):
        params = apply_lifting(system, lam)
        ctxt = system.construct_context(params, np.random.randint(1000))
        return local_propagation(samples, ctxt, local_md_config)

    def batch_log_prob(samples, lam):
        params = apply_lifting(system, lam)
        return -batch_u(samples, params)

    return batch_propagate, batch_log_prob


from timemachine.md.barostat.utils import compute_box_center


def _image_frames(group_idxs, ligand_idxs, frames, boxes) -> np.ndarray:
    assert np.array(boxes).shape[1:] == (3, 3), "Boxes are not 3x3"
    assert len(frames) == len(boxes), "Number of frames and boxes don't match"
    imaged_frames = np.empty_like(frames)
    for i, (frame, box) in enumerate(zip(frames, boxes)):
        assert frame.ndim == 2 and frame.shape[-1] == 3, "frames must have shape (N, 3)"
        # Recenter the frame around the centroid of the ligand
        ligand_centroid = np.mean(frame[ligand_idxs], axis=0)
        center = compute_box_center(box)
        offset = ligand_centroid + center
        centered_frames = frame - offset

        imaged_frames[i] = image_frame(group_idxs, centered_frames, box)
    return np.array(imaged_frames)


def inplace_randomly_rotate(coords, ligand_idxs, jordan_idxs):
    ligand_coords = coords[ligand_idxs]
    # ligand_centroid = np.mean(coords[ligand_idxs], axis=0, keepdims=True)
    jordan_centroid = np.mean(coords[jordan_idxs], axis=0, keepdims=True)
    # remove centroid
    centered_coords = ligand_coords - jordan_centroid
    rotated_coords = np.matmul(centered_coords, special_ortho_group.rvs(3))
    coords[ligand_idxs] = rotated_coords + jordan_centroid


import pickle


def estimate_populations(mol, host_pdb, ff, outfile, n_walkers, n_burn_in_steps, n_eq_steps, n_relax_steps, seed):
    system = ComplexPhaseSystem(mol, host_pdb, ff)
    params_1 = apply_lifting(system, lam=1.0)

    # TODO: generate or load pre-generated samples
    print("Generating initial samples...")
    samples = system.generate_initial_samples(
        params_1, burn_in=n_burn_in_steps, n_samples=n_walkers, n_eq_steps=n_eq_steps, seed=seed
    )

    print("\trandomly rotating the ligand")
    for sample in samples:
        # assume that the protein doesn't move very much
        # tbd: do we rotate ligand's velocities as well?
        inplace_randomly_rotate(sample.coords, system.ligand_idxs, system.jordan_idxs)

    local_md_config = LocalMDConfig(
        n_relax_steps,  # TODO: INCREASE THIS n_equilibrium_steps between runs
        system.ligand_idxs.astype(np.int32),  # local_idxs
        0,  # burn_in
        0,  # store_x_interval
        radius=2.5,  # 25 Angstrom LMD radius
        k=200,
    )

    batch_propagate, batch_log_prob = make_smc_funcs(
        system, local_md_config=local_md_config  # of steps we re-run in equilibrium
    )

    bond_idxs = system.potentials[0].idxs
    group_idxs = get_group_indices(bond_idxs.tolist(), len(system.masses))

    def write_frames_callback(iteration, lamb, log_weights, cur_samples, final, path):
        weights = np.exp(log_weights - logsumexp(log_weights))
        print(f"iteration {iteration} lamb {lamb}")
        print(f"weights {weights.tolist()}")
        print(f"log_weights {log_weights.tolist()}")
        frames = _image_frames(
            group_idxs,
            system.ligand_idxs,
            [c.coords for c in cur_samples],
            [c.box for c in cur_samples],
        )

        if final:
            out_path = path + "_final.cif"
        else:
            out_path = path + "_" + str(iteration) + ".cif"

        print(f"writing out samples to {out_path}")
        writer = cif_writer.CIFWriter([system.complex_topology, mol], out_path)
        for frame in frames:
            writer.write_frame(frame * 10)
        writer.close()

    # forward
    log_weights = np.zeros(len(samples))

    # (ytz) see how reversible the processes are without introducing charge changes

    # coupling
    lambda_0_samples, fwd_lambdas, fwd_log_weights_traj = adaptive_neq_switch(
        samples,
        batch_propagate,
        batch_log_prob,
        functools.partial(resample_fxn, thresh=0.0),
        log_weights,
        initial_lam=1.0,
        final_lam=0.5,  # REVERTME
        max_num_lambdas=5000,
        callback_fn=functools.partial(write_frames_callback, path=outfile + "_fwd"),
        callback_interval=40,
    )

    print("fwd_lambdas", fwd_lambdas)
    print("log_weights_fwd", fwd_log_weights_traj[-1])

    # keep old weights
    log_weights = fwd_log_weights_traj[-1]
    resampled = lambda_0_samples

    # Alternatively, we can re-sample, and then set weights back to zero
    # print("Forcing re-sampling for the reverse process")
    # indices, log_weights = resample_fxn(fwd_log_weights_traj[-1], thresh=1.0)
    # resampled = [lambda_0_samples[i] for i in indices]

    # decoupling using samples and log_weights from the coupling process
    lambda_1_samples, rev_lambdas, rev_log_weights_traj = adaptive_neq_switch(
        resampled,
        batch_propagate,
        batch_log_prob,
        functools.partial(resample_fxn, thresh=0.0),
        log_weights,
        initial_lam=0.5,  # REVERTME
        final_lam=1.0,
        max_num_lambdas=5000,
        callback_fn=functools.partial(write_frames_callback, path=outfile + "_rev"),
        callback_interval=40,
    )

    print("log_weights_rev", rev_log_weights_traj[-1])

    with open(outfile + "_traj.pkl", "wb") as fh:
        data = [
            lambda_0_samples,
            fwd_lambdas,
            fwd_log_weights_traj,
            lambda_1_samples,
            rev_lambdas,
            rev_log_weights_traj,
        ]
        pickle.dump(data, fh)

    # t3 = time()

    # log_weights = result_dict["log_weights_traj"][-1]
    # df = -one_sided_exp(-log_weights)

    # msg = f"""
    # done in a total of {t3 - t0:.3f} s
    #     initialization, energy minimization: {t1 - t0:.3f} s
    #     generating end-point samples from decoupled waterbox + ligand: {t2 - t1:.3f} s
    #     applying adaptive SMC: {t3 - t2:.3f} s
    # """

    # results = dict(
    #     times=(t1 - t0, t2 - t1, t3 - t2),
    #     timing_message=msg,
    #     df=df,
    #     log_weights_traj=result_dict["log_weights_traj"],
    #     incremental_log_weights_traj=result_dict["incremental_log_weights_traj"],
    #     lambdas=result_dict["lambdas"],
    # )

    # return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Swimming Simulations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--ligand", type=str, help="ligand sdf", required=True)
    parser.add_argument("--protein", type=str, help="protein pdb", required=True)
    parser.add_argument("--outfile", type=str, help="outfile pdb", required=True)
    parser.add_argument("--n_walkers", type=int, help="num walkers", required=True)
    parser.add_argument(
        "--n_burn_in_steps", type=int, help="num burn in steps at the starting end-state", required=True
    )
    parser.add_argument("--n_eq_steps", type=int, help="num steps to take between drawing samples", required=True)
    parser.add_argument("--n_relax_steps", type=int, help="relaxation time between moving in lambda", required=True)
    parser.add_argument("--seed", type=int, help="rng seed", required=True)

    cmd_args = parser.parse_args()

    suppl = [m for m in Chem.SDMolSupplier(cmd_args.ligand, removeHs=False)]
    mol = suppl[0]
    protein_pdb = cmd_args.protein

    ff = Forcefield.load_default()

    estimate_populations(
        mol,
        protein_pdb,
        ff,
        cmd_args.outfile,
        cmd_args.n_walkers,
        cmd_args.n_burn_in_steps,
        cmd_args.n_eq_steps,
        cmd_args.n_relax_steps,
        cmd_args.seed,
    )

    # tbd: flat-bottom restraint, turn it off alchemicall
    # tbd: decharge ligand first, then turn it back on
