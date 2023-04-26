# implement local HMC using the selection mask feature added in https://github.com/proteneer/timemachine/pull/1005
# (and also implement global HMC for comparison)

import numpy as np
from jax import jit

from timemachine.constants import BOLTZ, DEFAULT_TEMP
from timemachine.lib import VelocityVerletIntegrator, custom_ops
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


def sample_a_selection_mask(i, x, box, radius=1.2, k=1000.0, temperature=DEFAULT_TEMP):
    r = distance_from_one_to_others(x[i], x, box)
    U_r = (k / 4) * (r > radius) * (r - radius) ** 4
    p_r = np.exp(-U_r / (BOLTZ * temperature))
    selection_mask = np.random.rand(len(r)) < p_r
    selection_mask[i] = False
    return selection_mask


def convert_to_idxs(selection_mask):
    return np.nonzero(selection_mask)[0].astype(np.int32)


class LocalHMC:
    def __init__(self, bound_potentials, dt, masses, temperature):
        self.dt = dt
        self.masses = masses
        self.temperature = temperature

        # integrator
        vv = VelocityVerletIntegrator(dt, masses)
        intg_impl = vv.impl()

        # potential energy function
        bound_impls = [p.to_gpu(np.float32).bound_impl for p in bound_potentials]
        self._params = [p.params for p in bound_potentials]
        self._flat_params = np.hstack([params.flatten() for params in self._params])
        ubps = [bound_potential.potential for bound_potential in bound_potentials]
        self.U_fxn = SummedPotential(ubps, self._params).bind(self._flat_params).to_gpu(np.float32)
        self._params = [p.params for p in bound_potentials]

        # need to initialize context with some values for x, v, box
        n = len(masses)
        x0 = np.random.randn(n, 3)
        v0 = np.random.randn(n, 3)
        box = np.eye(3) * 5
        ctxt = custom_ops.Context(x0, v0, box, intg_impl, bound_impls)
        self.ctxt = ctxt

    def _sample_velocities(self):
        n = len(self.masses)
        v_unscaled = np.random.randn(n, 3)

        sigma = np.sqrt(BOLTZ * self.temperature) * np.sqrt(1 / self.masses)
        v_scaled = v_unscaled * np.expand_dims(sigma, axis=1)

        assert v_scaled.shape == (n, 3)

        return v_scaled

    def _reduced_kinetic_energy(self, v) -> float:
        return 0.5 * np.sum((self.masses[:, np.newaxis] * (v ** 2))) / (BOLTZ * self.temperature)

    def _reduced_total_energy(self, U, v) -> float:
        u = U / (BOLTZ * self.temperature)
        ke = self._reduced_kinetic_energy(v)
        return u + ke

    def propose(self, xvb: CoordsVelBox, n_steps=100, i=0, radius=1.2, k=1000.0):
        x0 = xvb.coords
        v0 = self._sample_velocities()  # ignore the velocities in CoordsVelBox
        box0 = xvb.box

        self.ctxt.set_x_t(x0)
        self.ctxt.set_v_t(v0)
        self.ctxt.set_box(box0)

        U_0 = self.U_fxn(x0, box0)
        log_prob_0 = -self._reduced_total_energy(U_0, v0)

        selection_mask = sample_a_selection_mask(i, x0, box0, radius, k, self.temperature)
        selection_idxs = convert_to_idxs(selection_mask)

        _, _ = self.ctxt.multiple_steps_local_selection(
            n_steps=n_steps,
            reference_idx=i,
            selection_idxs=selection_idxs,
            store_x_interval=0,
            radius=radius,
            k=k,
        )

        x_prop = self.ctxt.get_x_t()
        v_prop = self.ctxt.get_v_t()
        U_prop = self.U_fxn(x_prop, box0)
        log_prob_prop = -self._reduced_total_energy(U_prop, v_prop)

        proposal = CoordsVelBox(x_prop, None, box0)
        return proposal, np.clip(log_prob_prop - log_prob_0, a_min=-np.inf, a_max=0)


class GlobalHMC:
    def __init__(self, bound_potentials, dt, masses, temperature):
        self.dt = dt
        self.masses = masses
        self.temperature = temperature

        # integrator
        vv = VelocityVerletIntegrator(dt, masses)
        intg_impl = vv.impl()

        # potential energy function
        bound_impls = [p.to_gpu(np.float32).bound_impl for p in bound_potentials]
        self._params = [p.params for p in bound_potentials]
        self._flat_params = np.hstack([params.flatten() for params in self._params])
        ubps = [bound_potential.potential for bound_potential in bound_potentials]
        self.U_fxn = SummedPotential(ubps, self._params).bind(self._flat_params).to_gpu(np.float32)
        self._params = [p.params for p in bound_potentials]

        # need to initialize context with some values for x, v, box
        n = len(masses)
        x0 = np.random.randn(n, 3)
        v0 = np.random.randn(n, 3)
        box = np.eye(3) * 5
        ctxt = custom_ops.Context(x0, v0, box, intg_impl, bound_impls)
        self.ctxt = ctxt

    def _reduced_kinetic_energy(self, v) -> float:
        return 0.5 * np.sum((self.masses[:, np.newaxis] * (v ** 2))) / (BOLTZ * self.temperature)

    def _reduced_total_energy(self, U, v) -> float:
        u = U / (BOLTZ * self.temperature)
        ke = self._reduced_kinetic_energy(v)
        return u + ke

    def propose(self, xvb: CoordsVelBox, n_steps=100):
        x0 = xvb.coords
        v0 = sample_maxwell_boltzmann(self.masses, self.temperature)  # ignores the velocities in CoordsVelBox
        box0 = xvb.box

        self.ctxt.set_x_t(x0)
        self.ctxt.set_v_t(v0)
        self.ctxt.set_box(box0)

        U_0 = self.U_fxn(x0, box0)
        log_prob_0 = -self._reduced_total_energy(U_0, v0)

        _, _ = self.ctxt.multiple_steps(
            n_steps=n_steps,
            store_x_interval=0,
        )

        x_prop = self.ctxt.get_x_t()
        v_prop = self.ctxt.get_v_t()
        U_prop = self.U_fxn(x_prop, box0)
        log_prob_prop = -self._reduced_total_energy(U_prop, v_prop)

        proposal = CoordsVelBox(x_prop, None, box0)
        return proposal, np.clip(log_prob_prop - log_prob_0, a_min=-np.inf, a_max=0)
