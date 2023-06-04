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
            self.jordan_idxs,
            self.rest_idxs,
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


def apply_lifting(host_system, lam=0.0):
    params = deepcopy(host_system.params)
    nb_params = params[-1]

    cutoff = 1.2
    switch = 0.5
    min_epsilon = 0.05

    if lam < switch:
        ligand_w_coords = 0
        ligand_charges = -2 * nb_params[host_system.ligand_idxs, 0] * (lam - switch)
        ligand_epsilons = np.clip(-2 * nb_params[host_system.ligand_idxs, 2] * (lam - switch), min_epsilon, np.inf)

        rest_w_coords = 0
        rest_charges = -2 * nb_params[host_system.rest_idxs, 0] * (lam - switch)
        rest_epsilons = np.clip(-2 * nb_params[host_system.rest_idxs, 2] * (lam - switch), min_epsilon, np.inf)
    else:
        ligand_w_coords = 2 * (lam - switch) * cutoff  # LIGAND atoms approach from positive w
        ligand_charges = 0.0
        ligand_epsilons = min_epsilon

        rest_w_coords = -2 * (lam - switch) * cutoff  # REST side chains approach from negative w
        rest_charges = 0.0
        rest_epsilons = min_epsilon

    nb_params = nb_params.at[host_system.ligand_idxs, -1].set(ligand_w_coords)
    nb_params = nb_params.at[host_system.ligand_idxs, 0].set(ligand_charges)
    nb_params = nb_params.at[host_system.ligand_idxs, 2].set(ligand_epsilons)

    nb_params = nb_params.at[host_system.rest_idxs, -1].set(rest_w_coords)
    nb_params = nb_params.at[host_system.rest_idxs, 0].set(rest_charges)
    nb_params = nb_params.at[host_system.rest_idxs, 2].set(rest_epsilons)

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


from functools import partial

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
    select_next_lam: NextLamSelector,
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
    select_next_lam
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

        # keep history, incompatible with NEQ for certain thresholds
        # lam_target = select_next_lam(cur_samples, log_weights, lam_initial)
        # forget history
        if final_lam == 0.0:
            direction = "fwd"
        elif final_lam == 1.0:
            direction = "rev"
        else:
            assert 0

        lam_target = select_next_lam_NEQ(t + 1, direction=direction)
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
    # with open(outfile + "_final.npz", "wb") as fh:
    # np.savez(fh, log_weights=log_weights, samples=cur_samples)

    # final result: a collection of samples, with associated log weights
    # incremental_log_weights = log_prob(samples, lambdas[-1]) - log_prob(samples, lambdas[-2])
    # incremental_log_weights_traj.append(incremental_log_weights)
    # log_weights_traj.append(np.array(log_weights + incremental_log_weights))

    print("completed lambda schedule", lambdas)


from scipy.optimize import root_scalar


def select_next_lam_CESS(
    samples,
    log_weights,
    current_lam,
    batch_log_prob,
    target_lam=0.0,
    frac_ess_reduction_threshold=0.02,
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
    # next_lam = np.clip(next_lam, current_lam, target_lam)

    return next_lam


def select_next_lam_NEQ(idx, direction):
    precomputed = [
        1.0,
        0.6863736973507093,
        0.6532195518540924,
        0.6303588779826937,
        0.6247279545443869,
        0.6212475732899223,
        0.6184643246202008,
        0.6165270044758466,
        0.6148755510845425,
        0.6133598983597282,
        0.6120365610546665,
        0.610866274308365,
        0.609804636091636,
        0.6088090273358695,
        0.6079395261140413,
        0.6070815289151585,
        0.6061550467831166,
        0.6053679286184421,
        0.6045832106254384,
        0.6038249628009691,
        0.6030339739365123,
        0.6023628938241303,
        0.6016817599388005,
        0.6010626634378666,
        0.6004523472377649,
        0.5998175907882101,
        0.5992295263616304,
        0.5985733924433865,
        0.5979766431581052,
        0.5973481932920566,
        0.5967927172857416,
        0.5962384962839901,
        0.595638180682657,
        0.5950434732531371,
        0.594515974557166,
        0.5939632384579835,
        0.593456383281639,
        0.5929080014771656,
        0.5923829463017837,
        0.5918694809631846,
        0.5913487374484712,
        0.5908312150142104,
        0.5903597348420847,
        0.5898680722903046,
        0.5893544587440602,
        0.5889057275035983,
        0.5884193017667141,
        0.5879516294062069,
        0.5874577290134941,
        0.5869963652214798,
        0.5865335586378227,
        0.5860862902451371,
        0.5856427880598759,
        0.5852414459509505,
        0.5848336804804382,
        0.5844125822883334,
        0.5839738620383356,
        0.583556522205564,
        0.583142746866567,
        0.5827233599890594,
        0.5822949439048658,
        0.58188822792123,
        0.5814869794936406,
        0.5810723284468395,
        0.5806844789043637,
        0.5802965903000248,
        0.5798822489951496,
        0.5794716481688293,
        0.5790662130795948,
        0.5786703881695169,
        0.5783155704782962,
        0.577934875080163,
        0.5775497704366684,
        0.577166868637966,
        0.5768129890336074,
        0.5764279248730232,
        0.5760667546652337,
        0.5756705559908025,
        0.5753128719740502,
        0.5749261293351712,
        0.5745722405721516,
        0.5742020156495034,
        0.5738264914062737,
        0.5734753825110146,
        0.5731086844141284,
        0.5727345552350284,
        0.5723656671856654,
        0.5719879910747818,
        0.5715973540602595,
        0.5712235311970738,
        0.5708468924202682,
        0.5704725298118314,
        0.570081115692323,
        0.5697211178833311,
        0.5693483121669415,
        0.5689685052506618,
        0.5686247497306659,
        0.5682544182399035,
        0.5679167436667295,
        0.567520937994069,
        0.5671550655054118,
        0.5667537575887903,
        0.566371433791304,
        0.5659688088298556,
        0.5656175391824126,
        0.5652388099592465,
        0.5648544120326359,
        0.5644911239445354,
        0.5641073600191038,
        0.563708828575993,
        0.5633525990616748,
        0.5629670200646066,
        0.5625621940904707,
        0.5621793763746642,
        0.5617957941583019,
        0.5613954059252217,
        0.5609926806450761,
        0.5606356503294051,
        0.5602486867707657,
        0.5598462444912438,
        0.5594488023180305,
        0.5590486477563132,
        0.5586592110942302,
        0.5582674558295221,
        0.557880633148125,
        0.5574968135301259,
        0.5570724395147345,
        0.5566986375570219,
        0.5562912391411656,
        0.5558772268744469,
        0.5554735199862529,
        0.5550855226092836,
        0.554683151792099,
        0.5543018041477279,
        0.5539295395715216,
        0.5535463028124152,
        0.55313848032615,
        0.5526993807460856,
        0.5523264608680556,
        0.5519514463848697,
        0.551518847474309,
        0.5511157614468106,
        0.5506903604075206,
        0.5502696017637881,
        0.5498701422416788,
        0.5494451051169792,
        0.5490508775755126,
        0.5486399195098544,
        0.54821143179973,
        0.5477471547107536,
        0.5473181003630719,
        0.5468693667629695,
        0.5464136215610002,
        0.5459262786166234,
        0.545456397716448,
        0.5450317623425113,
        0.5445682503331737,
        0.544100258149093,
        0.5436340206276276,
        0.543157077545844,
        0.5426837246649654,
        0.5422403636317779,
        0.5417879454101912,
        0.5413533645960811,
        0.5408899305634399,
        0.5404110572044323,
        0.5399474955048461,
        0.5394762242326853,
        0.5390423705221978,
        0.5385751656066522,
        0.5380898849559653,
        0.5376064853850059,
        0.5371168879115108,
        0.5366301370607645,
        0.5361592497793924,
        0.5356746592785239,
        0.5351959586268903,
        0.5347330170537593,
        0.5342552677475078,
        0.5338183885115051,
        0.5334105486507147,
        0.5329480930207088,
        0.532468883843731,
        0.5319800685234376,
        0.5315455865257767,
        0.5310935712361532,
        0.5306414475676124,
        0.5301713774674168,
        0.5297077198705231,
        0.5292495547115943,
        0.5288064712861844,
        0.5283427002602636,
        0.5279424363375217,
        0.5274784453698423,
        0.5270589960042149,
        0.5265568339800673,
        0.5260540526515411,
        0.5255747595941963,
        0.5251103566199844,
        0.5246176528166695,
        0.5241142844005006,
        0.5236468606314114,
        0.5231055221652284,
        0.5226271729883684,
        0.5220702406244755,
        0.5215571225987259,
        0.5210735104803998,
        0.5205655183982553,
        0.5200028190386768,
        0.5194695876538956,
        0.51890413983653,
        0.5183559435551857,
        0.5177940500621733,
        0.5172312445097822,
        0.51666172651595,
        0.5160254161330668,
        0.5154321546394942,
        0.5148066141505255,
        0.5141085675161616,
        0.5133989220618987,
        0.5126702500930788,
        0.5118906126249836,
        0.5109916615046808,
        0.5101380264704146,
        0.5091962825655468,
        0.5080228079331245,
        0.5068731363392024,
        0.5053594258153796,
        0.5032466326147279,
        0.4967544247994828,
        0.4928482389939868,
        0.489034807436785,
        0.4852138227771676,
        0.4810159813706461,
        0.47650892172274445,
        0.474256137738313,
        0.4724509353009355,
        0.47116768025434363,
        0.4697101689519604,
        0.46839682849635145,
        0.46702264992994086,
        0.46544685198375074,
        0.46388384675994576,
        0.4622955041084296,
        0.4607012967791437,
        0.45909857085347644,
        0.45758925022637853,
        0.45625122601654655,
        0.4547532435815298,
        0.4532136417602865,
        0.45167939986053196,
        0.45000381814606943,
        0.4483461008800625,
        0.4467841651875904,
        0.44512307205754054,
        0.44336519112844436,
        0.4415043675086764,
        0.4396842931411113,
        0.437844217025921,
        0.4358337898719781,
        0.4338236583183168,
        0.4317219868412401,
        0.42970737637616285,
        0.42745442225684493,
        0.4253142963283444,
        0.42315819617681716,
        0.4209000586041037,
        0.4186268630800378,
        0.41619294848861765,
        0.4140402968495099,
        0.4117138472754145,
        0.4094413510693857,
        0.4071174011915065,
        0.404674140501017,
        0.40217010851811513,
        0.3999072546649204,
        0.39779409878896127,
        0.3956930961693831,
        0.39351108326474094,
        0.39136749116441694,
        0.38924743311926707,
        0.3871107673264959,
        0.3848442972985366,
        0.3824888094447824,
        0.3801911338994137,
        0.37792843791104047,
        0.3755735416774162,
        0.3730041274202568,
        0.37071310276713587,
        0.3681960158242726,
        0.3658827123301742,
        0.36323987871675323,
        0.36073532663714825,
        0.35813974693932316,
        0.3555668791087785,
        0.3528407063174647,
        0.35015528724371653,
        0.34737711236674496,
        0.34453442268468865,
        0.34184939145746346,
        0.3390988493910195,
        0.3363770095618628,
        0.33382385520180946,
        0.33125389373323033,
        0.328582558409749,
        0.3256253921015641,
        0.3229001209743109,
        0.31997026832542924,
        0.3170122931736741,
        0.3141144215406968,
        0.31115113356848006,
        0.30836020962010857,
        0.30525699551338814,
        0.3022264214133788,
        0.2993026339551649,
        0.29625198949135173,
        0.2933533273841218,
        0.29028033207787646,
        0.28732332975237207,
        0.28387047055485054,
        0.280868296971382,
        0.27752920072993076,
        0.27428749799930113,
        0.2711385848389753,
        0.2678200427929861,
        0.26428037687369565,
        0.2609276303341306,
        0.2577972568736997,
        0.2544321449956842,
        0.2513306275386505,
        0.24791366460935976,
        0.2449202406003017,
        0.24158805203238015,
        0.23821200939955123,
        0.23480138364770725,
        0.23138215366058001,
        0.22789694786684056,
        0.22467795693913492,
        0.22138611187283241,
        0.21818484165378005,
        0.21480885531839916,
        0.2115906152549063,
        0.20819320246934667,
        0.2049427755865634,
        0.20168800316822239,
        0.19843034071877197,
        0.1953635952655758,
        0.19196643249603654,
        0.1885322827309734,
        0.1853946201541292,
        0.18199094049295567,
        0.17875779627264948,
        0.17553276385126954,
        0.1723038155432551,
        0.16902718726975482,
        0.16579603847418503,
        0.16236352248194635,
        0.1591952766918687,
        0.1558920003702621,
        0.15271679274108135,
        0.14935997902784484,
        0.1458532275150143,
        0.14241584356066658,
        0.13923538177688546,
        0.1358870619632048,
        0.1327471471027431,
        0.12929865389010547,
        0.12579971817242577,
        0.12242823752720475,
        0.11910654246537726,
        0.11564583430299515,
        0.11223659809112865,
        0.10902452177216501,
        0.1056491030371182,
        0.10229495213242522,
        0.09879480615495921,
        0.09532746629933438,
        0.09213252586046325,
        0.08879488614968678,
        0.08538229531214878,
        0.08186971358043792,
        0.07861631572758503,
        0.07548836743100815,
        0.0723068719532095,
        0.06947578132271774,
        0.06625683996918212,
        0.06317960925413166,
        0.0597760402427705,
        0.05629818031983482,
        0.052901897120737236,
        0.04974714641045044,
        0.046676649421543974,
        0.04342443149991788,
        0.040168843415938134,
        0.0371408540394345,
        0.034154470686494645,
        0.03088610723614219,
        0.027984074348763825,
        0.02509867336068908,
        0.022208798826096776,
        0.019428296524711886,
        0.016585149832086976,
        0.013577574056999907,
        0.010766025533898327,
        0.007968936582912846,
        0.005046187497380326,
        0.0022804244722839347,
        0.0,
    ]
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

    select_next_lam = partial(select_next_lam_CESS, batch_log_prob=batch_log_prob)
    # select_next_lam = partial(select_next_lam_NEQ, batch_log_prob=batch_log_prob)

    return batch_propagate, batch_log_prob, select_next_lam


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

    batch_propagate, batch_log_prob, select_next_lam = make_smc_funcs(
        system, local_md_config=local_md_config  # of steps we re-run in equilibrium
    )

    bond_idxs = system.potentials[0].idxs
    group_idxs = get_group_indices(bond_idxs.tolist(), len(system.masses))

    def write_frames_callback(iteration, lamb, log_weights, cur_samples, final):
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
            out_path = outfile + "_final.cif"
        else:
            out_path = outfile + "_" + str(iteration) + ".cif"

        print(f"writing out samples to {out_path}")
        writer = cif_writer.CIFWriter([system.complex_topology, mol], out_path)
        for frame in frames:
            writer.write_frame(frame * 10)
        writer.close()

    # forward
    log_weights = np.zeros(len(samples))

    # coupling
    lambda_0_samples, fwd_lambdas, fwd_log_weights_traj = adaptive_neq_switch(
        samples,
        select_next_lam,
        batch_propagate,
        batch_log_prob,
        functools.partial(resample_fxn, thresh=0.0),
        log_weights,
        initial_lam=1.0,
        final_lam=0.0,
        max_num_lambdas=1000,
        callback_fn=write_frames_callback,
        callback_interval=20,
    )

    print("log_weights_fwd", fwd_log_weights_traj[-1])

    # keep old weights
    # log_weights = fwd_log_weights_traj[-1]

    # Alternatively, we can re-sample, and then set weights back to zero
    print("Forceing re-sampling for the reverse process")
    indices, log_weights = resample_fxn(fwd_log_weights_traj[-1], thresh=1.0)
    resampled = [lambda_0_samples[i] for i in indices]

    # decoupling using samples and log_weights from the coupling process
    lambda_1_samples, rev_lambdas, rev_log_weights_traj = adaptive_neq_switch(
        resampled,
        select_next_lam,
        batch_propagate,
        batch_log_prob,
        functools.partial(resample_fxn, thresh=0.0),
        log_weights,
        initial_lam=0.0,
        final_lam=1.0,
        max_num_lambdas=1000,
        callback_fn=write_frames_callback,
        callback_interval=20,
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
