from jax.config import config

config.update("jax_enable_x64", True)

import timemachine

print(timemachine.__version__)

import argparse
from copy import deepcopy
from time import time

import numpy as np

from timemachine import lib
from timemachine.constants import BOLTZ, DEFAULT_TEMP
from timemachine.fe.model_utils import apply_hmr
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

        self.potentials, self.params, self.physical_masses, self.coords, self.box = enhanced.get_complex_phase_system(
            mol, host_pdb, ff, minimize_energy=True
        )

        num_total_atoms = len(self.coords)
        num_ligand_atoms = self.mol.GetNumAtoms()
        self.ligand_idxs = np.array(np.arange(num_total_atoms - num_ligand_atoms, num_total_atoms), dtype=np.int32)
        assert len(self.ligand_idxs) == self.mol.GetNumAtoms()

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

        masses = apply_hmr(self.physical_masses, bond_list)

        intg = lib.LangevinIntegrator(temperature, dt, friction, masses, seed)
        intg_impl = intg.impl()

        group_idxs = get_group_indices(bond_list)
        barostat_interval = 5

        barostat = lib.MonteCarloBarostat(len(masses), pressure, temperature, group_idxs, barostat_interval, seed + 1)
        barostat_impl = barostat.impl(all_impls)

        ctxt = custom_ops.Context(
            self.coords, np.zeros_like(self.coords), self.box, intg_impl, all_impls, barostat_impl
        )

        # ugg, otherwise code that returns only `ctxt` will segfault...
        ctxt._all_impls = all_impls
        ctxt._barostat = barostat
        ctxt._barostat_impl = barostat_impl
        ctxt._intg = intg
        ctxt._intg_impl = intg_impl

        return ctxt

    def generate_initial_samples(
        self,
        params,
        n_samples=500,
        seed=2022,
        burn_in=10_000,
    ):
        """bind self.potentials to params, construct contexts, run simulation"""

        print(f"generating samples (seed={seed})")
        ctxt = self.construct_context(params, seed)

        # burn-in
        print(f"\tburning in for {burn_in} steps...")
        t0 = time()
        _ = ctxt.multiple_steps(burn_in)
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


def apply_lifting(solvent_system, lam=0.0):
    params = deepcopy(solvent_system.params)
    nb_params = params[-1]
    nb_params = nb_params.at[solvent_system.ligand_idxs, -1].set(lam)
    return tuple(params[:-1]) + (nb_params,)


@dataclass
class LocalMDConfig:
    n_steps: int
    local_idxs: np.array
    burn_in: int
    store_x_interval: int
    radius: float
    k: float


def _local_propagate(xvb, ctxt, local_md_config, seed):
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
    return CoordsVelBox(x, v, box)


# TODO: do all subsequent steps of local propagation with a fixed selection mask?


def local_propagation(xvbs, ctxt, local_md_config):
    updates = []

    for xvb in xvbs:
        seed = np.random.randint(10000)

        updates.append(_local_propagate(xvb, ctxt, local_md_config, seed))

    return updates


# def global_propagation(xvbs, ctxt):


def global_propagation(xvbs, ctxt, n_steps):
    samples = []
    for xvb in xvbs:
        ctxt.set_box(xvb.box)
        ctxt.set_v_t(xvb.velocities)
        ctxt.set_x_t(xvb.coords)

        ctxt.multiple_steps(n_steps)

        x = ctxt.get_x_t()
        v = ctxt.get_v_t()
        box = ctxt.get_box()
        xvb = CoordsVelBox(x, v, box)
        samples.append(xvb)

    return samples


from functools import partial

# now, convert from kBT to whatever unit dG is in -- kJ/mol?
from timemachine.fe.reweighting import one_sided_exp
from timemachine.md.smc import conditional_multinomial_resample, effective_sample_size

# how to do summed potential again?
from timemachine.potentials import SummedPotential


def resample_fxn(log_weights):
    ess = effective_sample_size(log_weights)
    msg = f"""
        ESS = {ess:.3f} ({100 * ess / len(log_weights):.3f}%)
        running estimate = {one_sided_exp(-log_weights):.3f}
    """
    print(msg)
    out = conditional_multinomial_resample(log_weights, thresh=0.5)

    return out


from typing import Callable

from timemachine.md.smc import BatchLogProb, BatchPropagator, Lambda, LogWeights, Resampler, ResultDict, Samples

NextLamSelector = Callable[[Samples, LogWeights, Lambda], Lambda]


def adaptive_sequential_monte_carlo(
    samples: Samples,
    select_next_lam: NextLamSelector,
    propagate: BatchPropagator,
    log_prob: BatchLogProb,
    resample: Resampler,
    initial_lam: Lambda = 1.0,
    final_lam: Lambda = 0.0,
    max_num_lambdas: int = 100,
    outfile=None,
) -> ResultDict:
    """barebones implementation of Sequential Monte Carlo (SMC)

    Parameters
    ----------
    samples: [N,] list
    select_next_lam
    propagate: function
        [move(x, lam) for x in xs]
        for example, move(x, lam) might mean "run 100 steps of all-atom MD targeting exp(-u(., lam)), initialized at x"
    log_prob: function
        [exp(-u(x, lam)) for x in xs]
    resample: function
        (optionally) perform resampling given an array of log weights

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

    n = len(samples)
    log_weights = np.zeros(n)

    # store
    # sample_traj = [samples]
    cur_samples = samples
    ancestry_traj = [np.arange(n)]
    log_weights_traj = [np.array(log_weights)]
    incremental_log_weights_traj = []  # note: redundant but convenient
    lambdas = [initial_lam]

    def accumulate_results(indices, log_weights, incremental_log_weights, lam):
        # sample_traj.append(samples)
        ancestry_traj.append(indices)
        log_weights_traj.append(np.array(log_weights))
        incremental_log_weights_traj.append(np.array(incremental_log_weights))
        lambdas.append(lam)

    lam_initial = lambdas[-1]
    lam_target = select_next_lam(cur_samples, log_weights, lam_initial)

    for t in range(max_num_lambdas):
        if lambdas[-1] == final_lam:
            break

        lam_initial = lambdas[-1]
        lam_target = select_next_lam(cur_samples, log_weights, lam_initial)

        # update log weights
        incremental_log_weights = log_prob(cur_samples, lam_target) - log_prob(cur_samples, lam_initial)
        log_weights += incremental_log_weights

        # print("normed weights", np.exp(log_weights - logsumexp(log_weights)))

        with open(outfile + "_" + str(t) + ".npz", "wb") as fh:
            np.savez(fh, log_weights=log_weights, samples=cur_samples)

        # resample
        indices, log_weights = resample(log_weights)
        resampled = [cur_samples[i] for i in indices]

        # propagate
        cur_samples = propagate(resampled, lam_target)

        # log
        accumulate_results(indices, log_weights, incremental_log_weights, lam_target)

        print(t, lambdas[-1])

    with open(outfile + "_final.npz", "wb") as fh:
        np.savez(fh, log_weights=log_weights, samples=cur_samples)

    # final result: a collection of samples, with associated log weights
    incremental_log_weights = log_prob(samples, lambdas[-1]) - log_prob(samples, lambdas[-2])
    incremental_log_weights_traj.append(incremental_log_weights)
    log_weights_traj.append(np.array(log_weights + incremental_log_weights))

    print("completed")


from scipy.optimize import root_scalar


def select_next_lam_CESS(
    samples,
    log_weights,
    current_lam,
    batch_log_prob,
    target_lam=0.0,
    frac_ess_reduction_threshold=0.05,
    xtol=1e-5,
    verbose=False,
):
    # CESS
    # implementation adapted from https://github.com/proteneer/timemachine/pull/442

    # TODO: phrase thresh multiplicatively instead of additively
    # (ESS_next = beta * ESS, rather than
    #  ESS_next = ESS - thresh)
    # as in "SMC with transformations" and prior work

    ess = effective_sample_size(log_weights)
    frac_ess = ess / len(log_weights)

    assert frac_ess >= frac_ess_reduction_threshold

    log_p_0 = batch_log_prob(samples, current_lam)

    # note scaled:
    # current_lam + 1.0 * remainder = target_lam
    direction = target_lam - current_lam

    def fractional_ess_as_fxn_of_increment(increment: float) -> float:
        trial_lam = current_lam + direction * increment
        log_p_trial = batch_log_prob(samples, trial_lam)
        incremental_log_weight = log_p_trial - log_p_0
        trial_log_weights = log_weights + incremental_log_weight
        trial_ess = effective_sample_size(trial_log_weights) / len(trial_log_weights)
        if verbose:
            print(increment, trial_ess)
        return np.nan_to_num(trial_ess, nan=0.0)

    def f(lam_increment: float) -> float:

        frec_ess_reduction = frac_ess - fractional_ess_as_fxn_of_increment(lam_increment)

        return frec_ess_reduction - frac_ess_reduction_threshold

    # try-except to catch rootfinding ValueError: f(a) and f(b) must have different signs
    #   which occurs when jumping all the way to lam=1.0 is still less than threshold
    try:
        result = root_scalar(f, bracket=(0, 1.0), xtol=xtol, maxiter=20)
        lam_increment = result.root
    except ValueError as e:
        print(f"root finding error: {e}")
        lam_increment = 1.0

    next_lam = current_lam + direction * lam_increment

    return next_lam


def make_smc_funcs(system, n_steps):

    U_fn = SummedPotential(system.potentials, system.params).to_gpu(np.float32).call_with_params_list

    def u(x, box, params):
        return U_fn(x, params, box) / kBT

    def batch_u(samples, params):
        return np.array([u(s.coords, s.box, params) for s in samples])

    def batch_global_propagate(samples, lam):
        params = apply_lifting(system, lam)
        ctxt = system.construct_context(params, np.random.randint(1000))
        return global_propagation(samples, ctxt, n_steps)

    def batch_log_prob(samples, lam):
        params = apply_lifting(system, lam)
        return -batch_u(samples, params)

    select_next_lam = partial(select_next_lam_CESS, batch_log_prob=batch_log_prob)

    return batch_global_propagate, batch_log_prob, select_next_lam


from scipy.stats import special_ortho_group


def inplace_randomly_translate_and_rotate(coords, ligand_idxs, box_width):
    ligand_coords = coords[ligand_idxs]
    offset = np.mean(ligand_coords, axis=0, keepdims=True)
    centered_coords = ligand_coords - offset
    rotated_coords = np.matmul(centered_coords, special_ortho_group.rvs(3))
    translated_coords = rotated_coords + np.random.randn(1, 3) * np.diag(box_width)
    coords[ligand_idxs] = translated_coords


def estimate_populations(mol, host_pdb, ff, outfile, n_walkers):

    np.random.seed(0)

    system = ComplexPhaseSystem(mol, host_pdb, ff)
    params_1 = apply_lifting(system, lam=1.0)

    # TODO: generate or load pre-generated samples
    samples = system.generate_initial_samples(params_1, n_samples=n_walkers)  # REVERTME: n_samples=100
    ligand_idxs = len(samples[0].coords) - np.arange(mol.GetNumAtoms()) - 1

    # resample ligands by randomly translating and rotating
    for sample in samples:
        inplace_randomly_translate_and_rotate(sample.coords, ligand_idxs, sample.box)

    # REVERTME: get local MD working again
    # local_md_config = LocalMDConfig(local_md_n_steps, system.ligand_idxs.astype(np.int32), 0, 0, local_md_radius, local_md_k)

    batch_global_propagate, batch_log_prob, select_next_lam = make_smc_funcs(system, n_steps=500)

    adaptive_sequential_monte_carlo(
        samples,
        select_next_lam,
        batch_global_propagate,
        batch_log_prob,
        resample_fxn,
        initial_lam=1.0,
        final_lam=0.0,
        max_num_lambdas=1000,
        outfile=outfile,
    )

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
    parser.add_argument("--num_walkers", type=int, help="num walkers", required=True)

    cmd_args = parser.parse_args()

    suppl = [m for m in Chem.SDMolSupplier(cmd_args.ligand, removeHs=False)]
    mol = suppl[0]
    protein_pdb = cmd_args.protein

    ff = Forcefield.load_default()

    estimate_populations(mol, protein_pdb, ff, cmd_args.outfile, cmd_args.num_walkers)
