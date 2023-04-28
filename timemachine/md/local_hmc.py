import numpy as np
from jax import jit

from timemachine.constants import BOLTZ, DEFAULT_TEMP
from timemachine.lib import VelocityVerletIntegrator, custom_ops
from timemachine.md.moves import MonteCarloMove
from timemachine.md.states import CoordsVelBox
from timemachine.potentials import SummedPotential
from timemachine.potentials.jax_utils import distance_from_one_to_others

distance_from_one_to_others = jit(distance_from_one_to_others)


def sample_maxwell_boltzmann(masses, temperature):
    n = len(masses)
    v_unscaled = np.random.randn(n, 3)

    sigma = np.sqrt(BOLTZ * temperature) * np.sqrt(1 / masses)
    v_scaled = v_unscaled * np.expand_dims(sigma, axis=1)

    assert v_scaled.shape == (n, 3)

    return v_scaled


def flat_bottom_U_r(r, k, radius):
    # needs to be consistent with definition in custom_ops
    return (k / 4) * (r > radius) * (r - radius) ** 4


def sample_a_selection_mask(i, x, box, radius=1.2, k=1000.0, temperature=DEFAULT_TEMP):
    """assumes flat-bottom quartic restraint"""
    r = distance_from_one_to_others(x[i], x, box)

    U_r = flat_bottom_U_r(r, k, radius)

    p_r = np.exp(-U_r / (BOLTZ * temperature))
    selection_mask = np.random.rand(len(r)) < p_r
    selection_mask[i] = False

    return selection_mask


def convert_to_idxs(selection_mask):
    return np.nonzero(selection_mask)[0].astype(np.int32)


class LocalHMC(MonteCarloMove):
    def __init__(self, bound_potentials, dt, masses, temperature):
        self.dt = dt
        self.masses = masses
        self.temperature = temperature

        # leapfrog integrator
        vv = VelocityVerletIntegrator(dt, masses)
        intg_impl = vv.impl()

        # potential energy function
        bound_impls = [p.to_gpu(np.float32).bound_impl for p in bound_potentials]
        self._params = [p.params for p in bound_potentials]
        self._flat_params = np.hstack([params.flatten() for params in self._params])
        ubps = [bound_potential.potential for bound_potential in bound_potentials]
        self.U_fxn = SummedPotential(ubps, self._params).bind(self._flat_params).to_gpu(np.float32)

        # need to initialize context with some values for x, v, box
        n = len(masses)
        x0 = np.random.randn(n, 3)
        v0 = np.random.randn(n, 3)
        box = np.eye(3) * 5
        ctxt = custom_ops.Context(x0, v0, box, intg_impl, bound_impls)
        self.ctxt = ctxt

    def _sample_velocities(self):
        n = len(self.masses)
        velocities = sample_maxwell_boltzmann(self.masses, self.temperature)

        assert velocities.shape == (n, 3)
        return velocities

    def _reduced_kinetic_energy(self, velocities) -> float:
        # input in physical units, output in reduced units
        kinetic_energy = 0.5 * np.sum((self.masses[:, np.newaxis] * (velocities ** 2)))
        return kinetic_energy / (BOLTZ * self.temperature)

    def _reduced_total_energy(self, potential_energy, velocities) -> float:
        # input in physical units, output in reduced units
        u = potential_energy / (BOLTZ * self.temperature)
        ke = self._reduced_kinetic_energy(velocities)
        return u + ke

    def propose(self, xvb: CoordsVelBox, n_steps=100, i=0, radius=1.2, k=1000.0):
        # augment with velocities
        x0 = xvb.coords
        v0 = self._sample_velocities()  # ignore the velocities in CoordsVelBox
        box0 = xvb.box
        self.ctxt.set_x_t(x0)
        self.ctxt.set_v_t(v0)
        self.ctxt.set_box(box0)

        # augment with selection mask
        selection_mask = sample_a_selection_mask(i, x0, box0, radius, k, self.temperature)

        # note! potential energy in accept/reject step
        # needs to include contribution from restraint
        r0 = distance_from_one_to_others(x0[i], x0, box0)
        U_restraint = np.sum(flat_bottom_U_r(r0, k, radius)[selection_mask])
        U0 = self.U_fxn(x0, box0) + U_restraint
        log_prob_0 = -self._reduced_total_energy(U0, v0)

        # perform local leapfrog on selected idxs, with appropriate restraint
        _, _ = self.ctxt.multiple_steps_local_selection(
            n_steps=n_steps,
            reference_idx=i,
            selection_idxs=convert_to_idxs(selection_mask),
            store_x_interval=0,
            radius=radius,
            k=k,
        )

        # get state after leapfrog
        x_prop = self.ctxt.get_x_t()
        v_prop = self.ctxt.get_v_t()
        box_prop = self.ctxt.get_box()
        assert np.all(box0 == box_prop)
        proposal = CoordsVelBox(x_prop, None, box_prop)

        # note! potential energy in accept/reject step
        # needs to include contribution from restraint
        r_prop = distance_from_one_to_others(x_prop[i], x_prop, box_prop)
        U_restraint_prop = np.sum(flat_bottom_U_r(r_prop, k, radius)[selection_mask])
        U_prop = self.U_fxn(x_prop, box_prop) + U_restraint_prop
        log_prob_prop = -self._reduced_total_energy(U_prop, v_prop)

        log_accept_prob = np.clip(log_prob_prop - log_prob_0, a_min=-np.inf, a_max=0)

        return proposal, log_accept_prob
