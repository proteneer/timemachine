"""
* Expand signatures to accept lam argument
    (mover.move(x) -> mover.move(x, lam))
* Do some magic to play nice with process parallelism
    (MoveImpl, move.initialize_once)
"""

import os
from typing import Tuple

import numpy as np

from timemachine import lib
from md.barostat.utils import get_bond_list, get_group_indices
from timemachine.lib import custom_ops
from md.states import CoordsVelBox


class MonteCarloMove:
    """Same as moves.MonteCarloMove, but accepting `lam` as an argument"""

    n_proposed: int = 0
    n_accepted: int = 0

    def propose(self, x: CoordsVelBox, lam: float) -> Tuple[CoordsVelBox, float]:
        """return proposed state and log acceptance probability"""
        raise NotImplementedError

    def move(self, x: CoordsVelBox, lam: float) -> CoordsVelBox:
        proposal, log_acceptance_probability = self.propose(x, lam)
        self.n_proposed += 1

        alpha = np.random.rand()
        acceptance_probability = np.exp(log_acceptance_probability)
        if alpha < acceptance_probability:
            self.n_accepted += 1
            return proposal
        else:
            return x

    @property
    def acceptance_fraction(self):
        if self.n_proposed > 0:
            return self.n_accepted / self.n_proposed
        else:
            return 0.0


class MoveImpl:
    """Solely for the purposes of process parallelism"""

    def __init__(self, bound_impls, barostat_impl, integrator_impl):
        self.bound_impls = bound_impls
        self.barostat_impl = barostat_impl
        self.integrator_impl = integrator_impl


class NPTMove(MonteCarloMove):
    def __init__(
        self,
        ubps,
        masses,
        temperature,
        pressure,
        n_steps,
        seed,
        dt=1.5e-3,
        friction=1.0,
        barostat_interval=5,
    ):
        print("constructing a new mover!")

        self.ubps = ubps
        self.masses = masses
        self.temperature = temperature
        self.pressure = pressure
        self.seed = seed
        self.dt = dt
        self.friction = friction
        self.barostat_interval = barostat_interval

        # intg = lib.LangevinIntegrator(temperature, dt, friction, masses, seed)
        # self.integrator_impl = intg.impl()
        # all_impls = [bp.bound_impl(np.float32) for bp in ubps]

        bond_list = get_bond_list(ubps[0])
        self.group_idxs = get_group_indices(bond_list)

        # barostat = lib.MonteCarloBarostat(len(masses), pressure, temperature, group_idxs, barostat_interval, seed + 1)
        # barostat_impl = barostat.impl(all_impls)

        # self.bound_impls = all_impls
        # self.barostat_impl = barostat_impl

        self.integrator_impl = None
        self.barostat_impl = None
        self.move_impl = None
        self.n_steps = n_steps

    def initialize_once(self):
        if self.move_impl is None:

            if "CUDA_VISIBLE_DEVICES" in os.environ:
                print("Initializing on:", os.environ["CUDA_VISIBLE_DEVICES"])
            else:
                print("initialize_once() called serially")

            bound_impls = [bp.bound_impl(np.float32) for bp in self.ubps]
            intg_impl = lib.LangevinIntegrator(self.temperature, self.dt, self.friction, self.masses, self.seed).impl()
            barostat_impl = lib.MonteCarloBarostat(
                len(self.masses), self.pressure, self.temperature, self.group_idxs, self.barostat_interval, self.seed + 1
            ).impl(bound_impls)
            self.move_impl = MoveImpl(bound_impls, barostat_impl, intg_impl)

        # else do nothing

    def propose(self, x: CoordsVelBox, lam: float):

        self.initialize_once()
        # note: context creation overhead here is actually very small!

        # print('impl', self.move_impl)
        ctxt = custom_ops.Context(
            x.coords,
            x.velocities,
            x.box,
            self.move_impl.integrator_impl,
            self.move_impl.bound_impls,
            self.move_impl.barostat_impl,
        )

        # arguments: lambda_schedule, du_dl_interval, x_interval
        _ = ctxt.multiple_steps(lam * np.ones(self.n_steps), 0, 0)
        x_t = ctxt.get_x_t()
        v_t = ctxt.get_v_t()
        box = ctxt.get_box()

        after_npt = CoordsVelBox(x_t, v_t, box)
        log_accept_prob = 0.0  # always accept

        return after_npt, log_accept_prob
