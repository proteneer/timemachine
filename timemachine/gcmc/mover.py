from jax import config

config.update("jax_enable_x64", True)

import jax
import numpy as np
from scipy.stats import special_ortho_group

from timemachine.constants import AVOGADRO, DEFAULT_KT, DEFAULT_TEMP
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import custom_ops
from timemachine.md import builders
from timemachine.potentials import nonbonded

CHEMICAL_POTENTIAL = 26.61236542799211  # calibrated using 12 window, staged AHFE
STANDARD_VOLUME = 0.029884155244114276  # calibrated using end-states of the above


def randomly_rotate_and_translate(coords, offset):
    centroid = np.mean(coords, axis=0, keepdims=True)
    centered_coords = coords - centroid
    rotated_coords = np.matmul(centered_coords, special_ortho_group.rvs(3))
    return rotated_coords + offset


class GCMCMover:
    def __init__(self, on_params, off_params, nb_beta, nb_cutoff):
        self.on_params = on_params
        self.off_params = off_params
        self.nb_beta = nb_beta
        self.nb_cutoff = nb_cutoff
        # counters
        self.i_count = 0
        self.d_count = 0
        self.t_count = 0

        @jax.jit
        def U_fn(conf, box, params, a_idxs, b_idxs):
            conf_i = conf[a_idxs]
            conf_j = conf[b_idxs]
            params_i = params[a_idxs]
            params_j = params[b_idxs]
            return nonbonded.nonbonded_block(conf_i, conf_j, box, params_i, params_j, self.nb_beta, self.nb_cutoff)

        self.U_fn = U_fn

    def toggle_gcmc_move(self, coords, box, cur_params, water_idxs, water_states):
        # this is a little bit different from the Adams formulation since we don't compute the 1/(N+1) or the N
        # prefactors.

        # 1. randomly turn off/on a water with 50/50 probability
        #   a. if turning off, compute delta_U when non-interacting
        #   b. if turning on, randomly rotate and translate, then turn on and compute delta_Us

        # print("num waters", np.sum(water_states))

        # TBD: should we compute cur_params from water_states?

        n_waters = len(water_states)
        n_atoms = len(coords)

        chosen_water = np.random.randint(0, n_waters)
        chosen_water_atoms = water_idxs[chosen_water]

        a_idxs = chosen_water_atoms
        b_idxs = np.setdiff1d(np.arange(n_atoms), chosen_water_atoms)

        self.t_count += 1

        box_vol = np.product(np.diag(box))

        beta = 1 / DEFAULT_KT
        adams_B = beta * CHEMICAL_POTENTIAL + np.log(box_vol / STANDARD_VOLUME)

        # useful for visualization
        TRANSLATIONAL_OFFSET = 0

        if water_states[chosen_water]:
            # turn interacting water off

            trial_params = cur_params.copy()

            np.testing.assert_equal(trial_params[chosen_water_atoms], self.on_params)

            # compute acceptance probability, the - sign is because we're missing
            # these interactions
            delta_U = -self.U_fn(coords, box, trial_params, a_idxs, b_idxs)
            # print("delta_U delete", delta_U)
            # assert delta_U < 1e4

            # p_delete = min(1, np.exp(-adams_B) * np.exp(-beta * delta_U))
            p_delete = min(1, np.exp(-adams_B - beta * delta_U))
            trial_params[chosen_water_atoms] = self.off_params

            if p_delete > np.random.rand():
                # move accepted
                next_water_state = water_states.copy()
                next_water_state[chosen_water] = 0
                # print("deletion success")
                self.d_count += 1
                trial_coords = coords.copy()
                trial_coords[chosen_water_atoms] += np.ones(3) * TRANSLATIONAL_OFFSET
                return trial_coords, trial_params, next_water_state
            else:
                # move rejected
                return coords, cur_params, water_states

        else:
            # turn non-interacting water on

            trial_params = cur_params.copy()
            trial_params[chosen_water_atoms] = self.on_params

            # randomly rotate and translate the chosen water
            trial_chosen_coords = coords[chosen_water_atoms]
            trial_translation = np.diag(box) * np.random.rand(3) + TRANSLATIONAL_OFFSET

            # tbd - velocities?
            moved_coords = randomly_rotate_and_translate(trial_chosen_coords, trial_translation)
            trial_coords = coords.copy()
            trial_coords[chosen_water_atoms] = moved_coords

            delta_U = self.U_fn(trial_coords, box, trial_params, a_idxs, b_idxs)

            # p_insert = min(1, np.exp(adams_B) * np.exp(-beta * delta_U))
            p_insert = min(1, np.exp(adams_B - beta * delta_U))
            if p_insert > np.random.rand() and np.abs(delta_U) < 1e4:
                # move accepted
                next_water_state = water_states.copy()
                next_water_state[chosen_water] = 1
                self.i_count += 1
                return trial_coords, trial_params, next_water_state
            else:
                # move rejected
                return coords, cur_params, water_states


def propagate_with_params(xi, vi, boxi, paramsi, masses, bps, seed, nsteps):
    temperature = DEFAULT_TEMP
    friction = 1.0
    dt = 1e-3

    intg_impl = custom_ops.LangevinIntegrator(masses, temperature, dt, friction, seed)

    bps[-1].params = paramsi
    bp_impls = [p.to_gpu(np.float32).bound_impl for p in bps]
    ctxt = custom_ops.Context(xi, vi, boxi, intg_impl, bp_impls)
    ctxt.multiple_steps_U(nsteps, 0, 0)

    return ctxt.get_x_t(), ctxt.get_v_t(), ctxt.get_box()


def compute_density(n_waters, box):
    box_vol = np.product(np.diag(box))
    numerator = n_waters * 18.01528 * 1e27
    denominator = box_vol * AVOGADRO * 1000
    return numerator / denominator


from timemachine.fe import cif_writer


def test_gcmc():
    # test GCMC water moves, targetted and untargetted, one water at a time, multiple waters at a time
    # first test: fill an empty box of water and assert equilibrium densities

    # setup a box of non-interacting water

    box_width = 3.0
    ff = Forcefield.load_default()
    host_system, host_coords, host_box, host_top = builders.build_water_system(box_width, ff.water_ff)

    # pad box by 1A to deal with jank
    # padding disabled since we're doing full insertions
    # host_box += np.eye(3) * 0.1

    print("Host Box", host_box)
    print("Expected # waters", np.product(np.diag(host_box)) / STANDARD_VOLUME)

    water_idxs = []
    # hack
    for atom_idx in range(len(host_coords) // 3):
        water_idxs.append([atom_idx * 3 + 0, atom_idx * 3 + 1, atom_idx * 3 + 2])

    water_idxs = np.array(water_idxs)
    print("N total waters", len(water_idxs))

    # 1 - interacting
    # 0 - non-interacting
    # water_states = np.ones(len(water_idxs), dtype=np.bool_)
    water_states = np.zeros(len(water_idxs), dtype=np.bool_)

    bps, masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)
    reference_params = bps[-1].params.copy()
    initial_params = reference_params

    nb_beta = bps[-1].potential.beta
    nb_cutoff = bps[-1].potential.cutoff

    off_params = np.zeros(4, dtype=np.float64)
    on_params = initial_params[[0, 1, 2]]

    mover = GCMCMover(on_params, off_params, nb_beta, nb_cutoff)

    cur_params = np.zeros_like(initial_params)  # important to sync with water_states

    cur_x_t = host_coords
    # cur_v_t = np.zeros_like(cur_x_t)
    cur_box = host_box

    # seed = 2023

    writer = cif_writer.CIFWriter([host_top], "water.cif")

    # nb_U_impl = bps[-1].potential.to_gpu(np.float32)

    # print("I", mover.i_count, "D", mover.d_count, "T", mover.t_count, "W", num_waters, "D", density)
    # for idx in range(2000):
    for idx in range(1000):
        # print(nb_U_impl(cur_x_t, cur_params, cur_box))
        # assert 0

        num_waters = np.sum(water_states)
        density = compute_density(num_waters, cur_box)

        print("I", mover.i_count, "D", mover.d_count, "T", mover.t_count, "W", num_waters, "D", density)
        writer.write_frame(cur_x_t * 10)

        for _ in range(5000):
            assert np.amax(np.abs(cur_x_t)) < 1e3
            cur_x_t, cur_params, water_states = mover.toggle_gcmc_move(
                cur_x_t,
                cur_box,
                cur_params,
                water_idxs,
                water_states,
            )

        # what do we do with velocities?
        # running MD is *optional*

        # writer.close()
        # assert 0
        # print(cur_x_t)
        # print(cur_box)
        # print(cur_params)
        # cur_x_t, cur_v_t, cur_box = propagate_with_params(
        # cur_x_t, cur_v_t, cur_box, cur_params, masses, bps, seed + idx, nsteps=5000
        # )

    # toggle state of a molecule by setting qij, eij, sij
    # print(reference_params)

    assert 0


if __name__ == "__main__":
    test_gcmc()
