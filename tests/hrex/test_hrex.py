import hypothesis.strategies as st
import numpy as np
from hypothesis import given, seed

from timemachine.md.hrex import get_samples_by_iter_by_replica


@given(
    st.integers(1, 10)
    .flatmap(lambda n_states: st.lists(st.permutations(range(n_states)), min_size=1, max_size=10))
    .map(np.array)
)
@seed(2023)
def test_get_samples_by_iter_by_replica(perms):
    n_iters, n_states = perms.shape
    replica_idx_by_state_by_iter = perms

    # NOTE: the implementation is agnostic to the instantiation of Samples; here we test with Samples = int
    # (replica, iter) -> samples
    samples_by_iter_by_replica_ref = np.arange(n_states * n_iters).reshape((n_states, n_iters))

    # (iter, replica) -> samples
    samples_by_replica_by_iter = samples_by_iter_by_replica_ref.swapaxes(0, 1)

    # (iter, state) -> samples
    samples_by_state_by_iter = np.take_along_axis(samples_by_replica_by_iter, replica_idx_by_state_by_iter, 1)

    samples_by_iter_by_replica = get_samples_by_iter_by_replica(
        samples_by_state_by_iter.tolist(), replica_idx_by_state_by_iter.tolist()
    )

    np.testing.assert_array_equal(samples_by_iter_by_replica, samples_by_iter_by_replica_ref)
