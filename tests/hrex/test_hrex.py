import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, seed

from timemachine.md.hrex import ReplicaIdx, get_normalized_kl_divergence, get_samples_by_iter_by_replica

pytestmark = [pytest.mark.nogpu]


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


def test_get_samples_by_iter_by_replica_invalid_args():
    with pytest.raises(AssertionError):
        get_samples_by_iter_by_replica([[1]], [[ReplicaIdx(0)], [ReplicaIdx(0)]])
    with pytest.raises(AssertionError):
        get_samples_by_iter_by_replica([[1], [2, 3]], [[ReplicaIdx(0)], [ReplicaIdx(0)]])
    with pytest.raises(AssertionError):
        get_samples_by_iter_by_replica([[1, 2], [3, 4]], [[ReplicaIdx(0)], [ReplicaIdx(0), ReplicaIdx(1)]])


def simulate_perfect_mixing_hrex(num_states: int, num_frames: int) -> list[list[int]]:
    """assume every step of HREX perfectly mixed all replicas"""
    inds: list[int] = np.arange(num_states).tolist()  # type: ignore
    traj = []
    for _ in range(num_frames):
        np.random.shuffle(inds)
        traj.append(list(inds))
    return traj


def simulate_slow_mixing_hrex(num_states: int, num_frames: int) -> list[list[int]]:
    """assume every step of HREX only succeeds in making num_states nearest-neighbor swaps"""
    traj: list[list[int]] = [np.arange(num_states).tolist()]  # type: ignore
    for _ in range(num_frames - 1):
        current_state = list(traj[-1])
        for _ in range(num_states):
            i = np.random.randint(num_states - 1)
            j = i + 1

            x_i = current_state[i]
            x_j = current_state[j]

            current_state[j] = x_i
            current_state[i] = x_j

        traj.append(current_state)
    return traj


def simulate_bottlenecked_hrex(num_states: int, num_frames: int) -> list[list[int]]:
    """simulate_slow_mixing_hrex, but there's a state near num_states/2 that never swaps with neighbors"""
    traj: list[list[int]] = [np.arange(num_states).tolist()]  # type: ignore

    bottleneck_i = int(round(num_states / 2))
    for _ in range(num_frames - 1):
        current_state = list(traj[-1])
        for _ in range(num_states):
            i = np.random.randint(num_states - 1)
            j = i + 1

            if i != bottleneck_i:
                x_i = current_state[i]
                x_j = current_state[j]

                current_state[j] = x_i
                current_state[i] = x_j

        traj.append(current_state)
    return traj


def simulate_no_mixing_hrex(num_states: int, num_frames: int) -> list[list[int]]:
    traj: list[list[int]] = [np.arange(num_states).tolist()] * num_frames  # type:ignore
    return traj


@pytest.mark.parametrize("seed", [2024])
@pytest.mark.parametrize("n_windows,frames", [(3, 2000), (16, 2000), (48, 2000)])
@pytest.mark.parametrize(
    "simulator",
    [
        simulate_perfect_mixing_hrex,
        simulate_perfect_mixing_hrex,
        simulate_slow_mixing_hrex,
        simulate_bottlenecked_hrex,
        simulate_no_mixing_hrex,
    ],
)
def test_normalized_kl_divergence(simulator, n_windows, frames, seed):
    """Verify that given any of the expected hrex simulations produces values that are greater than 0.0"""
    np.random.seed(seed)
    hrex_matrix = simulator(n_windows, frames)
    res = get_normalized_kl_divergence(hrex_matrix)
    assert res >= 0.0


@pytest.mark.parametrize("n_windows,frames", [(16, 2000), (48, 2000)])
def test_normalized_kl_divergence_perfect_mixing(n_windows, frames):
    hrex_matrix = simulate_perfect_mixing_hrex(n_windows, frames)
    res = get_normalized_kl_divergence(hrex_matrix)
    assert res >= 0.0
    np.testing.assert_allclose(res, 0.0, atol=0.015)


@pytest.mark.parametrize("n_windows,frames", [(16, 2000), (48, 2000)])
def test_normalized_kl_divergence_no_mixing(n_windows, frames):
    """With no mixing at all, the kl divergence becomes larger than 1.0 and continues to grow with the number of windows

    Verify that values are greater than one in this case.
    """
    hrex_matrix = simulate_no_mixing_hrex(n_windows, frames)
    res = get_normalized_kl_divergence(hrex_matrix)
    assert res >= 1.0


@pytest.mark.parametrize("n_windows,frames", [(5, 2000), (9, 2000), (16, 2000), (48, 2000)])
def test_normalized_kl_divergence_with_bottleneck(n_windows, frames):
    """With a bottleneck, expect the divergence to be between 0.5 and 1.0

    Note for n_windows <= 3 this test will fail as the assertion is unreliable for small numbers of windows"""
    hrex_matrix = simulate_bottlenecked_hrex(n_windows, frames)
    res = get_normalized_kl_divergence(hrex_matrix)
    assert res >= 0.5


@pytest.mark.parametrize("seed", [2024])
@pytest.mark.parametrize("iterations", [10])
@pytest.mark.parametrize("n_windows,frames", [(5, 2000), (16, 2000), (48, 2000)])
def test_normalized_kl_divergence_relative_values(iterations, n_windows, frames, seed):
    """Ensure that the mean of multiple replicates of normalized KL divergence produce the right ordering.

    Perfect sampling should have the lowest value, followed by slow mixing followed up by bottle necked mixing"""
    np.random.seed(seed)
    perfect_mixing_kl = []
    slow_mixing_kl = []
    bottlenecked_kl = []
    for _ in range(iterations):
        perfect_mixing_kl.append(get_normalized_kl_divergence(simulate_perfect_mixing_hrex(n_windows, frames)))
        bottlenecked_kl.append(get_normalized_kl_divergence(simulate_bottlenecked_hrex(n_windows, frames)))
        slow_mixing_kl.append(get_normalized_kl_divergence(simulate_slow_mixing_hrex(n_windows, frames)))

    assert np.mean(perfect_mixing_kl) < np.mean(slow_mixing_kl)
    assert np.mean(slow_mixing_kl) < np.mean(bottlenecked_kl)
