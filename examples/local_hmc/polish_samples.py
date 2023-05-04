# load langevin trajectories, apply a little Barker MCMC to them

import numpy as np
from jax import grad

from timemachine import constants
from timemachine.md.barker import BarkerProposal
from timemachine.md.states import CoordsVelBox
from timemachine.potentials import SummedPotential
from timemachine.testsystems.dhfr import setup_dhfr

dhfr_langevin_samples = np.load("dhfr_langevin_samples.npz")
xs, boxes = dhfr_langevin_samples["xs"], dhfr_langevin_samples["boxes"]
near_eq_samples = [CoordsVelBox(x, None, box) for (x, box) in zip(xs, boxes)]

# define energy function
host_fns, host_masses, host_conf, box = setup_dhfr(1.2)
_params = [p.params for p in host_fns]
_flat_params = np.hstack([params.flatten() for params in _params])
ubps = [bound_potential.potential for bound_potential in host_fns]
U_fxn = SummedPotential(ubps, _params).bind(_flat_params).to_gpu(np.float32)


def mcmc_propose(x, box, barker, temperature=constants.DEFAULT_TEMP):
    kBT = constants.BOLTZ * temperature

    x_prop = barker.sample(x)
    log_fwd = barker.log_density(x, x_prop)
    log_rev = barker.log_density(x_prop, x)
    log_prob_0 = -U_fxn(x, box) / kBT
    log_prob_prop = -U_fxn(x_prop, box) / kBT

    log_accept_prob = min(0.0, (log_rev - log_fwd) + (log_prob_prop - log_prob_0))

    return x_prop, log_accept_prob


def polish_sample(x, box, n_steps=100):
    temperature = constants.DEFAULT_TEMP
    kBT = constants.BOLTZ * temperature
    proposal_stddev = 0.0001

    def grad_log_q(x):
        return -grad(U_fxn)(x, box) / kBT

    barker_prop = BarkerProposal(grad_log_q, proposal_stddev)

    x = np.array(x)
    n_accept = 0
    for t in range(n_steps):

        x_prop, log_accept_prob = mcmc_propose(x, box, barker_prop)

        if np.random.rand() < np.exp(log_accept_prob):
            x = x_prop
            n_accept += 1

    print(f"acceptance fraction = {n_accept / n_steps}")

    return x


xs_polished = []
print(f"len(xs) = {len(xs)}")
for (x, box) in zip(xs, boxes):
    x_polished = polish_sample(x, box)
    xs_polished.append(x_polished)

np.savez("dhfr_mcmc_samples.npz", xs=np.array(xs_polished), boxes=boxes)
