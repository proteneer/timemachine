from functools import partial

import numpy as np
from scipy.special import logsumexp

from timemachine.md.gradual import ReversibleNCMCMove


# TODO: speed up, or pytest.mark slow if > 10 seconds
def test_ncmc_1d():
    """construct a bimodal distribution in 1D, and check that NCMC helps get unstuck from initial mode"""

    np.random.seed(0)

    def log_q(x, lam):
        """mixture of 2 gaussians:
        * when lam = 0, the components have loc = +/- 4
        * when lam = 1, the components have loc = 0
        """
        radius = 4 * (1 - lam)
        log_q_components = np.array([-((x - radius) ** 2), -((x + radius) ** 2)])
        log_q_mix = logsumexp(log_q_components)
        return log_q_mix

    # construct a lambda schedule that starts and end at equilibrium
    decouple_lam_schedule = np.linspace(0, 1, 50)
    recouple_lam_schedule = np.linspace(1, 0, 50)
    round_trip_lambda_schedule = np.hstack([decouple_lam_schedule, recouple_lam_schedule])
    T = len(round_trip_lambda_schedule)
    _start = round_trip_lambda_schedule[0]
    _end = round_trip_lambda_schedule[-1]
    assert (_start == 0.0) and (_start == _end), "must start and end at target"

    # construct propagators
    class RWMHPropagator:
        def __init__(self, lam):
            self.lam = lam

        def move(self, x):
            x_prop = x + np.random.randn()
            log_accept_prob = min(log_q(x_prop, self.lam) - log_q(x, self.lam), 0.0)
            accepted = np.random.rand() < np.exp(log_accept_prob)
            return x_prop if accepted else x

    propagators = [RWMHPropagator(lam) for lam in round_trip_lambda_schedule[1:]]
    assert len(propagators) == (T - 1), "api sharp edge, may revert: don't need to propagate 0th state"

    # construct functions to evaluate log_p[i](x) - log_p[i-1](x)
    def logpdf_difference(x, lam, lam_next):
        return log_q(x, lam_next) - log_q(x, lam)

    logpdf_difference_fxns = [
        partial(
            logpdf_difference,
            lam=round_trip_lambda_schedule[i],
            lam_next=round_trip_lambda_schedule[i + 1],
        )
        for i in range(T - 1)
    ]
    ncmc = ReversibleNCMCMove(propagators, logpdf_difference_fxns)

    ######

    # simulate 1000 ncmc moves
    n_cycles = 1000
    _traj = [-4.0]
    aux_traj = []
    for t in range(n_cycles):
        x_next, aux = ncmc.move(_traj[-1])

        _traj.append(x_next)
        aux_traj.append(aux)
    ncmc_traj = np.array(_traj)

    well_indicator_traj = ncmc_traj < 0

    # check for at least some productive moves
    num_transitions = (np.diff(well_indicator_traj) != 0).sum()
    assert num_transitions > 50  # arbitrary test threshold > 0

    # check for approximately equal time spent in the +4 and -4 wells
    fraction_in_starting_well = np.mean(ncmc_traj < 0)  # random trials: 0.49, 0.55, ...
    assert 0.4 < fraction_in_starting_well < 0.6

    ######

    # secondary assertion: the component RWMH propagator
    #   directly targeting lam=0 is really stuck in starting well
    base_prop = RWMHPropagator(0.0)

    _base_traj = [ncmc_traj[0]]
    for _ in range(len(ncmc_traj)):
        _base_traj.append(base_prop.move(_base_traj[-1]))
    base_traj = np.array(_base_traj)

    fraction_in_starting_well_base = np.mean(base_traj < 0)
    assert fraction_in_starting_well_base == 1.0

    # TODO: make sure the test thresholds aren't too brittle across a few random seeds
