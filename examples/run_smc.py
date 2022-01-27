import argparse
from datetime import datetime
from functools import partial
from pickle import dump

import numpy as np

from fe.free_energy_rabfe import construct_pre_optimized_absolute_lambda_schedule_solvent
from md.smc import simple_smc, conditional_multinomial_resample

from testsystems.biphenyl import construct_biphenyl_test_system

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_walkers", type=int, help="number of walkers", default=100)
    parser.add_argument("--n_windows", type=int, help="number of lambda windows", default=100)
    parser.add_argument("--n_md_steps", type=int, help="number of MD steps per move", default=1000)
    parser.add_argument("--resample_thresh", type=float, help="resample when fractional ESS < thresh", default=0.6)
    cmd_args = parser.parse_args()
    print(cmd_args)

    # SMC set up
    n_walkers = cmd_args.n_walkers
    n_windows = cmd_args.n_windows
    resample_thresh = cmd_args.resample_thresh
    n_md_steps = cmd_args.n_md_steps
    config = dict(n_walkers=n_walkers, n_windows=n_windows, resample_thresh=resample_thresh, n_md_steps=n_md_steps)

    potential_energy_fxn, mover, initial_samples = construct_biphenyl_test_system(n_steps=n_md_steps)

    sample_inds = np.random.choice(np.arange(len(initial_samples)), size=n_walkers)
    samples = [initial_samples[i] for i in sample_inds]
    lambdas = construct_pre_optimized_absolute_lambda_schedule_solvent(n_windows)[::-1]
    resample = partial(conditional_multinomial_resample, thresh=resample_thresh)

    def advance(xlam):
        x, lam = xlam
        mover.lamb = lam
        x_next = mover.move(x)

        return x_next

    def u(xlam):
        x, lam = xlam
        return potential_energy_fxn.u(x, lam)

    def propagate(xs, lam):
        xlams = [(x, lam) for x in xs]
        xs_next = map(advance, xlams)
        return xs_next

    def log_prob(xs, lam):
        xlams = [(x, lam) for x in xs]
        u_s = np.array(map(u, xlams))
        return -u_s

    smc_result = simple_smc(
        samples,
        lambdas,
        propagate,
        log_prob,
        resample,
    )

    with open(f"smc_result_{datetime.now()}.pkl", "wb") as f:
        dump((smc_result, config), f)

    traj, log_weights_traj, ancestry_traj, incremental_log_weights_traj = smc_result

# TODO: use less disk space!
