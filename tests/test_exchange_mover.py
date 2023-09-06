from jax import config

config.update("jax_enable_x64", True)

import numpy as np

from timemachine.constants import DEFAULT_KT
from timemachine.md.exchange import exchange_mover


def test_batch_log_weights_incremental():

    W = 111  # num waters
    N = 439  # num atoms
    nb_beta = 1.2
    nb_cutoff = 0.6
    nb_params = np.random.rand(N, 4)
    nb_params[:, 0] -= 0.5
    nb_params[:, 1] *= 0.01
    nb_params[:, -1] = 0
    beta = 1 / DEFAULT_KT

    water_idxs = []
    for wi in range(W):
        water_idxs.append([wi * 3 + 0, wi * 3 + 1, wi * 3 + 2])  # has to be contiguous

    bdem = exchange_mover.BDExchangeMove(nb_beta, nb_cutoff, nb_params, water_idxs, beta)

    for _ in range(100):
        conf = np.random.rand(N, 3)
        box = np.eye(3) * 5
        initial_weights = bdem.batch_log_weights(conf, box)
        water_idx = np.random.randint(W)

        old_pos = conf[water_idxs[water_idx]]
        new_pos = old_pos + np.random.rand(1, 3)
        test_log_weights, trial_coords = bdem.batch_log_weights_incremental(
            conf, box, water_idx, new_pos, initial_weights
        )

        new_conf = conf.copy()
        new_conf[water_idxs[water_idx]] = new_pos
        ref_final_weights = bdem.batch_log_weights(new_conf, box)

        np.testing.assert_allclose(trial_coords, new_conf)
        np.testing.assert_allclose(np.array(test_log_weights), np.array(ref_final_weights))
