import numpy as np

from timemachine.md.hrex import get_samples_by_iter_by_replica


def test_get_samples_by_iter_by_replica():
    rng = np.random.default_rng()
    n_states = 5
    n_iters = 10
    samples_per_iter = 2

    # (replica, iter) -> samples
    samples_by_iter_by_replica_ref = rng.uniform(0, 1, size=(n_states, n_iters, samples_per_iter))

    # (iter, state) -> replica
    replica_idx_by_state_by_iter = rng.permuted(np.repeat(np.arange(n_states)[None, :], n_iters, axis=0), axis=1)
    # (iter, replica) -> samples
    samples_by_replica_by_iter = samples_by_iter_by_replica_ref.swapaxes(0, 1)

    # (iter, state) -> samples
    samples_by_state_by_iter = np.take_along_axis(
        samples_by_replica_by_iter, replica_idx_by_state_by_iter[:, :, None], 1
    )

    samples_by_iter_by_replica = get_samples_by_iter_by_replica(
        samples_by_state_by_iter.tolist(), replica_idx_by_state_by_iter.tolist()
    )

    np.testing.assert_array_equal(samples_by_iter_by_replica, samples_by_iter_by_replica_ref)
