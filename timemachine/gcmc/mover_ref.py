# reference mover that does does not toggle, but explicitly tries insertion and deletion
from jax import config

config.update("jax_enable_x64", True)

import jax
import numpy as np
from scipy.stats import special_ortho_group

from timemachine.constants import AVOGADRO, DEFAULT_KT, DEFAULT_TEMP
from timemachine.fe import cif_writer, model_utils
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.gcmc import grand_utils
from timemachine.lib import custom_ops
from timemachine.md import builders
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.potentials import nonbonded

# CHEMICAL_POTENTIAL = -26.61236542799211  # calibrated using 12 windows, staged AHFE
CHEMICAL_POTENTIAL = -26.29842368687200  # calibrated using 24 windows, staged AHFE
STANDARD_VOLUME = 0.029884155244114276  # calibrated using end-states of the above
# desired densities:
# Average Density (E0) 1000.7382640866621 kg/m^3
# Average Density (E1) 1001.3561571327649 kg/m^3


def randomly_rotate_and_translate(coords, offset):
    # original GCMC code rotates on a random water
    centroid = np.mean(coords, axis=0, keepdims=True)
    centered_coords = coords - centroid
    rotated_coords = np.matmul(centered_coords, special_ortho_group.rvs(3))
    return rotated_coords + offset


class GCMCMover:
    def __init__(self, on_params, off_params, nb_beta, nb_cutoff, box, water_idxs):
        self.on_params = on_params
        self.off_params = off_params
        self.nb_beta = nb_beta
        self.nb_cutoff = nb_cutoff
        self.water_idxs = water_idxs
        # counters
        self.i_count = 0
        self.d_count = 0
        self.t_count = 0

        self.box = box
        self.beta = 1 / DEFAULT_KT
        self.adams_B = self.beta * CHEMICAL_POTENTIAL + np.log(np.product(np.diag(box)) / STANDARD_VOLUME)

        print("Adams Parameter", self.adams_B)

        @jax.jit
        def U_fn(conf, params, a_idxs, b_idxs):
            conf_i = conf[a_idxs]
            conf_j = conf[b_idxs]
            params_i = params[a_idxs]
            params_j = params[b_idxs]
            return nonbonded.nonbonded_block(conf_i, conf_j, box, params_i, params_j, self.nb_beta, self.nb_cutoff)

        self.U_fn = U_fn

    # super slow
    # def construct_params_from_states(self, water_states):
    #     # cur_params = np.zeros_like(initial_params)  # important to sync with water_states
    #     # tbd vectorize broadcast later
    #     params = np.zeros((len(water_states) * 3, 4))
    #     for mol_idx, state in enumerate(water_states):
    #         water_atoms = self.water_idxs[mol_idx]
    #         if state == 2 or state == 1:
    #             params[water_atoms] = self.on_params
    #         else:
    #             params[water_atoms] = self.off_params
    #     return params

    # faster, vectorized version
    def construct_params_from_states(self, water_states):
        n_waters = len(water_states)
        b_on_params = np.repeat(self.on_params[None, :], n_waters, axis=0).reshape(-1, 4)
        b_off_params = np.repeat(self.off_params[None, :], n_waters, axis=0).reshape(-1, 4)
        mask = np.repeat(water_states != 0, 3, axis=0)
        mask = np.expand_dims(mask, axis=1)
        return np.where(mask, b_on_params, b_off_params)

    def insertion_move(self, coords, water_states):
        # pick a random water that is not currently active.
        N_inactive_waters = np.sum(water_states == 0)
        if N_inactive_waters == 0:
            # we've saturated the number of ghost waters
            assert 0
            return coords, water_states
        n_atoms = len(coords)

        inactive_waters = np.argwhere(water_states == 0).reshape(-1)
        chosen_water = np.random.choice(inactive_waters)
        chosen_water_atoms = self.water_idxs[chosen_water]

        trial_params = self.construct_params_from_states(water_states)
        np.testing.assert_array_almost_equal(trial_params[chosen_water_atoms], self.off_params)
        trial_params[chosen_water_atoms] = self.on_params

        # randomly rotate and translate the chosen water
        trial_chosen_coords = coords[chosen_water_atoms]
        trial_translation = np.diag(self.box) * np.random.rand(3)
        # tbd - velocities?
        moved_coords = randomly_rotate_and_translate(trial_chosen_coords, trial_translation)
        trial_coords = coords.copy()
        trial_coords[chosen_water_atoms] = moved_coords

        a_idxs = chosen_water_atoms
        b_idxs = np.setdiff1d(np.arange(n_atoms), chosen_water_atoms)

        delta_U = self.U_fn(trial_coords, trial_params, a_idxs, b_idxs)

        N_active_waters = np.sum(water_states == 1)
        p_insert = min(1, np.exp(self.adams_B - self.beta * delta_U) / (N_active_waters + 1))

        if np.random.rand() < p_insert:
            # move accepted
            next_water_state = water_states.copy()
            # toggle water from off -> on
            assert next_water_state[chosen_water] == 0
            next_water_state[chosen_water] = 1
            self.i_count += 1
            return trial_coords, next_water_state
        else:
            # move rejected
            return coords, water_states

    def deletion_move(self, coords, water_states):
        N_active_waters = np.sum(water_states == 1)
        n_atoms = len(coords)

        if N_active_waters == 0:
            # reject deletion move
            return coords, water_states

        active_waters = np.argwhere(water_states == 1).reshape(-1)
        chosen_water = np.random.choice(active_waters)
        chosen_water_atoms = self.water_idxs[chosen_water]

        trial_params = self.construct_params_from_states(water_states)
        np.testing.assert_equal(trial_params[chosen_water_atoms], self.on_params)

        a_idxs = chosen_water_atoms
        b_idxs = np.setdiff1d(np.arange(n_atoms), chosen_water_atoms)

        delta_U = -self.U_fn(coords, trial_params, a_idxs, b_idxs)
        p_accept = min(1, N_active_waters * np.exp(-self.adams_B - self.beta * delta_U))

        if np.random.rand() < p_accept:
            # move accepted
            next_water_state = water_states.copy()
            assert next_water_state[chosen_water] == 1
            next_water_state[chosen_water] = 0
            self.d_count += 1
            return coords, next_water_state
        else:
            # move rejected
            return coords, water_states

    def gcmc_move(self, *args, **kwargs):
        self.t_count += 1
        if np.random.rand() < 0.5:
            return self.insertion_move(*args, **kwargs)
        else:
            return self.deletion_move(*args, **kwargs)


def propagate_with_params(xi, vi, boxi, paramsi, masses, bps, seed, nsteps):
    temperature = DEFAULT_TEMP
    friction = 1.0
    dt = 1e-3

    intg_impl = custom_ops.LangevinIntegrator(masses, temperature, dt, friction, seed)

    bps[-1].params = paramsi
    bp_impls = [p.to_gpu(np.float32).bound_impl for p in bps]
    ctxt = custom_ops.Context(xi, vi, boxi, intg_impl, bp_impls)
    ctxt.multiple_steps_U(nsteps, 0, 0)

    np.testing.assert_array_equal(ctxt.get_box(), boxi)

    return ctxt.get_x_t(), ctxt.get_v_t(), ctxt.get_box()


def compute_density(n_waters, box):
    box_vol = np.product(np.diag(box))
    numerator = n_waters * 18.01528 * 1e27
    denominator = box_vol * AVOGADRO * 1000
    return numerator / denominator


def image_frame(group_idxs, frame, box) -> np.ndarray:
    assert frame.ndim == 2 and frame.shape[-1] == 3, "frames must have shape (N, 3)"

    # move every molecule into the home box
    # centered_frames = frame - compute_box_center(box)
    return model_utils.image_frame(group_idxs, frame, box)


def offset_ghost_waters(frame, water_states, water_idxs):
    frame_copy = frame.copy()
    for mol_idx, state in enumerate(water_states):
        if state == 0:
            for atom_idx in water_idxs[mol_idx]:
                frame_copy[atom_idx] += np.ones(3) * 5
    return frame_copy


def test_gcmc():
    # test GCMC water moves, targetted and untargetted, one water at a time, multiple waters at a time
    # first test: fill an empty box of water and assert equilibrium densities

    # setup a box of non-interacting water
    box_width = 4.0
    ff = Forcefield.load_default()
    host_system, host_coords, host_box, host_top = builders.build_water_system(box_width, ff.water_ff)
    num_ghosts = 200
    host_system, host_coords, host_top = grand_utils.add_ghosts(host_top, host_coords, num_ghosts, ff.water_ff)
    print(host_top.getNumAtoms())
    print(host_coords.shape)

    print("Host Box", host_box)
    print("Expected # of waters", np.product(np.diag(host_box)) / STANDARD_VOLUME)

    water_idxs = []
    for atom_idx in range(len(host_coords) // 3):
        water_idxs.append([atom_idx * 3 + 0, atom_idx * 3 + 1, atom_idx * 3 + 2])

    water_idxs = np.array(water_idxs)
    print("N total waters available", len(water_idxs))

    # 0 - ghost, non-interacting
    # 1 - ghost, fully-interacting
    # 2 - non-ghost, fully-interacting
    # set everything to ghost
    water_states = np.zeros(len(host_coords) // 3, dtype=np.int32) + 1
    water_states[-num_ghosts:] = 0

    print("Initial # of non-ghost waters", np.sum(water_states == 2))
    print("Initial # of interacting ghost waters", np.sum(water_states == 1))
    print("Initial # of non-interacting ghost waters", np.sum(water_states == 0))

    bps, masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)
    reference_params = bps[-1].params.copy()
    initial_params = reference_params

    nb_beta = bps[-1].potential.beta
    nb_cutoff = bps[-1].potential.cutoff

    off_params = np.zeros((3, 4), dtype=np.float64)
    on_params = initial_params[[0, 1, 2]]

    cur_box = host_box

    mover = GCMCMover(on_params, off_params, nb_beta, nb_cutoff, cur_box, water_idxs)

    # initialize on and off params

    cur_x_t = host_coords
    cur_v_t = np.zeros_like(cur_x_t)

    # equilibrate
    seed = 2023
    cur_params = mover.construct_params_from_states(water_states)
    cur_x_t, cur_v_t, cur_box = propagate_with_params(
        cur_x_t, cur_v_t, cur_box, cur_params, masses, bps, seed - 1, nsteps=100000
    )

    writer = cif_writer.CIFWriter([host_top], "water.cif")

    group_idxs = get_group_indices(get_bond_list(bps[0].potential), len(host_coords))

    # GCMC/MD
    md_steps_per_batch = 1000
    for idx in range(50000):
        num_waters = np.sum(water_states != 0)
        num_ghost_waters = np.sum(water_states == 0)
        density = compute_density(num_waters, cur_box)

        # densities.append(density)
        cur_x_t = image_frame(group_idxs, cur_x_t, cur_box)

        if idx % 50 == 0:
            print(
                "I",
                mover.i_count,
                "D",
                mover.d_count,
                "T",
                mover.t_count,
                "W",
                num_waters,
                "G",
                num_ghost_waters,
                "D",
                density,
                "S",
                idx * md_steps_per_batch,
            )

        writer.write_frame(offset_ghost_waters(cur_x_t, water_states, water_idxs) * 10)

        for _ in range(1000):
            assert np.amax(np.abs(cur_x_t)) < 1e3
            cur_x_t, water_states = mover.gcmc_move(cur_x_t, water_states)

        # what do we do with velocities?
        # running MD is optional, but should in theory speed things up
        cur_params = mover.construct_params_from_states(water_states)
        cur_x_t, cur_v_t, cur_box = propagate_with_params(
            cur_x_t, cur_v_t, cur_box, cur_params, masses, bps, seed + idx, nsteps=md_steps_per_batch
        )


if __name__ == "__main__":
    test_gcmc()
