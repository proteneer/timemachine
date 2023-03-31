import hashlib

import networkx as nx
import numpy as np
import pytest
from scipy.stats import linregress

from timemachine.fe.mle import infer_node_vals, infer_node_vals_and_errs, infer_node_vals_and_errs_networkx

pytestmark = [pytest.mark.nogpu]


def generate_instance(g: nx.Graph, edge_noise_stddev=0.0):
    K = g.number_of_nodes()
    E = g.number_of_edges()

    # generate true node vals
    node_vals = np.random.randn(K)

    # read off true edge vals
    edge_idxs = np.array(g.edges)
    src_idxs, dst_idxs = edge_idxs.T
    true_edge_diffs = node_vals[dst_idxs] - node_vals[src_idxs]

    # observe edges, with noise
    noise = edge_noise_stddev * np.random.randn(E)
    obs_edge_diffs = true_edge_diffs + noise
    edge_stddevs = edge_noise_stddev * np.ones(E)

    return node_vals, edge_idxs, obs_edge_diffs, edge_stddevs


def generate_and_solve_instance(g: nx.Graph, edge_noise_stddev=0.0):
    node_vals, edge_idxs, obs_edge_diffs, edge_stddevs = generate_instance(g, edge_noise_stddev)

    # infer node vals
    est_node_vals = infer_node_vals(edge_idxs, obs_edge_diffs, edge_stddevs)

    # assert we recovered node_vals, ignoring arbitrary additive offset
    ground_truth = node_vals - node_vals[0]
    estimated = est_node_vals - est_node_vals[0]

    return ground_truth, estimated


def generate_random_regular_graph(i):
    # K * d must be even
    _K = np.random.randint(11, 500)
    d = np.random.randint(2, 10)
    K = _K + ((_K * d) % 2)
    g = nx.random_graphs.random_regular_graph(d=d, n=K, seed=hash(i * K * d))
    return g


def generate_random_valid_regular_graph(max_retries=10):
    n_retries = 0
    g = generate_random_regular_graph(n_retries)

    while nx.number_connected_components(g) != 1 and n_retries < max_retries:
        n_retries += 1
        g = generate_random_regular_graph(n_retries)
    return g


def test_random_spanning_graphs_with_no_edge_noise():
    np.random.seed(2022)
    for i in range(10):

        K = np.random.randint(3, 500)
        g = nx.random_tree(K, seed=hash(K * i))
        E = g.number_of_edges()
        assert E == K - 1
        print(f"instance: random tree on {K} nodes ({E} noiseless edges)")

        ground_truth, estimated = generate_and_solve_instance(g, edge_noise_stddev=0.0)

        print(f"\tmax |error| = {np.max(np.abs(ground_truth - estimated))}")

        # should be able to recover exactly
        np.testing.assert_allclose(ground_truth, estimated, atol=1e-5)


@pytest.mark.skip(reason="need better assert")
def test_random_spanning_graphs_with_some_edge_noise():
    np.random.seed(2022)
    for i in range(10):

        K = np.random.randint(3, 500)
        sigma = np.random.rand()
        g = nx.random_tree(K, seed=hash(K * i))
        E = g.number_of_edges()
        assert E == K - 1
        print(f"instance: random tree on {K} nodes\n\t{E} noisy edges, sigma\t\t= {sigma:.3f}")

        ground_truth, estimated = generate_and_solve_instance(g, edge_noise_stddev=sigma)

        residuals = ground_truth - estimated

        print(f"\tstddev(true - estimated)\t= {np.std(residuals):.3f}")

        # TBD: what's an informative assertion here?
        # * Probably need to do some error-propagation
        #   in terms of all-pairs shortest path lengths,
        #   not just in terms of E or K...
        assert np.std(residuals) < np.log(K) * sigma


def test_random_graphs_containing_cycles_and_no_edge_noise():
    np.random.seed(2022)

    for i in range(10):
        g = generate_random_valid_regular_graph()

        K = g.number_of_nodes()
        E = g.number_of_edges()
        d = E // K
        print(f"instance: random {d}-regular graph on {K} nodes ({E} noiseless edges)")

        ground_truth, estimated = generate_and_solve_instance(g, edge_noise_stddev=0.0)

        print(f"\tmax |error| = {np.max(np.abs(ground_truth - estimated))}")

        # should be able to recover exactly
        np.testing.assert_allclose(ground_truth, estimated, atol=1e-5)


def test_random_graphs_containing_cycles_with_edge_noise():
    np.random.seed(2022)

    for _ in range(10):
        g = generate_random_valid_regular_graph()

        K = g.number_of_nodes()
        E = g.number_of_edges()
        d = E // K
        sigma = np.random.rand()
        print(f"instance: random {d}-regular graph on {K} nodes\n\t{E} noisy edges, sigma\t\t= {sigma:.3f}")
        ground_truth, estimated = generate_and_solve_instance(g, edge_noise_stddev=sigma)

        residuals = ground_truth - estimated

        print(f"\tstddev(true - estimated)\t= {np.std(residuals):.3f}")

        # JF: TBD: what's an informative assertion here?
        # * Probably need to do some error-propagation
        #   in terms of all-pairs shortest path lengths,
        #   not just in terms of E or K...
        assert np.std(residuals) < np.log(K) * sigma


def test_infer_node_dgs_w_error():

    np.random.seed(0)

    for _ in range(5):
        edge_noise_stddev = np.random.rand()
        g = generate_random_valid_regular_graph()
        n_nodes = g.number_of_nodes()

        node_vals, edge_idxs, obs_edge_diffs, edge_stddevs = generate_instance(g, edge_noise_stddev)

        num_refs = np.random.randint(n_nodes)
        ref_node_idxs = np.random.choice(np.arange(n_nodes), num_refs, replace=False)
        ref_node_vals = node_vals[ref_node_idxs]
        ref_node_stddevs = 0.01 * np.ones(num_refs)

        dg, dg_err = infer_node_vals_and_errs(
            edge_idxs,
            obs_edge_diffs,
            edge_stddevs,
            ref_node_idxs,
            ref_node_vals,
            ref_node_stddevs,
            seed=np.random.randint(1000),
        )
        assert (dg_err > 0).all()

        res = linregress(dg, node_vals)
        assert res.rvalue > 0.9


def test_infer_node_vals_and_errs_networkx():

    np.random.seed(0)

    edge_noise_stddev = np.random.rand()
    g = generate_random_valid_regular_graph()
    g = nx.convert_node_labels_to_integers(g)
    n_nodes = g.number_of_nodes()

    node_vals, edge_idxs, obs_edge_diffs, edge_stddevs = generate_instance(g, edge_noise_stddev)

    num_refs = np.random.randint(n_nodes)
    ref_node_idxs = np.random.choice(np.arange(n_nodes), num_refs, replace=False)
    ref_node_vals = node_vals[ref_node_idxs]
    ref_node_stddevs = 0.01 * np.ones(num_refs)

    edge_diff_prop = "edge_diff"
    edge_stddev_prop = "edge_stddev"
    node_val_prop = "node_val"
    node_stddev_prop = "node_stddev"

    for e, diff, stddev in zip(g.edges.values(), obs_edge_diffs, edge_stddevs):
        e[edge_diff_prop] = diff
        e[edge_stddev_prop] = stddev

    for n, ref_val, ref_stddev in zip(ref_node_idxs, ref_node_vals, ref_node_stddevs):
        g.nodes[n][node_val_prop] = ref_val
        g.nodes[n][node_stddev_prop] = ref_stddev

    seed = np.random.randint(1000)

    ref_dgs, ref_dg_errs = infer_node_vals_and_errs(
        edge_idxs, obs_edge_diffs, edge_stddevs, ref_node_idxs, ref_node_vals, ref_node_stddevs, seed=seed
    )

    g_res = infer_node_vals_and_errs_networkx(
        g, edge_diff_prop, edge_stddev_prop, node_val_prop, node_stddev_prop, ref_node_idxs, seed=seed
    )

    for n, (ref_dg, ref_dg_err) in enumerate(zip(ref_dgs, ref_dg_errs)):
        assert g_res.nodes[n][node_val_prop] == ref_dg
        assert g_res.nodes[n][node_stddev_prop] == ref_dg_err

    # ensure results are invariant wrt relabeling nodes
    idx_to_label = {n: hashlib.sha256(b"{n}") for n in g.nodes}
    g = nx.relabel_nodes(g, idx_to_label)
    ref_node_labels = [idx_to_label[n] for n in ref_node_idxs]

    g_res = infer_node_vals_and_errs_networkx(
        g, edge_diff_prop, edge_stddev_prop, node_val_prop, node_stddev_prop, ref_node_labels, seed=seed
    )

    for n, (ref_dg, ref_dg_err) in enumerate(zip(ref_dgs, ref_dg_errs)):
        assert g_res.nodes[idx_to_label[n]][node_val_prop] == ref_dg
        assert g_res.nodes[idx_to_label[n]][node_stddev_prop] == ref_dg_err


def test_infer_node_vals_incorrect_sizes():
    """Verify that infer_node_vals correctly asserts that the length of arrays are the same"""
    g = generate_random_valid_regular_graph()
    n_nodes = g.number_of_nodes()

    node_vals, edge_idxs, obs_edge_diffs, edge_stddevs = generate_instance(g, 1.0)
    num_refs = np.random.randint(n_nodes)
    ref_node_idxs = np.random.choice(np.arange(n_nodes), num_refs, replace=False)
    ref_node_vals = node_vals[ref_node_idxs]

    with pytest.raises(AssertionError):
        infer_node_vals(
            edge_idxs,
            obs_edge_diffs,
            edge_stddevs,
            ref_node_idxs,
            [],
        )
    infer_node_vals(
        edge_idxs,
        obs_edge_diffs,
        edge_stddevs,
        ref_node_idxs,
        ref_node_vals,
    )


def test_infer_node_vals_and_errs_incorrect_sizes():
    """Verify that infer_node_vals_and_errs correctly asserts that the length of arrays are the same"""
    g = generate_random_valid_regular_graph()
    n_nodes = g.number_of_nodes()

    node_vals, edge_idxs, obs_edge_diffs, edge_stddevs = generate_instance(g, 1.0)
    num_refs = np.random.randint(n_nodes)
    ref_node_idxs = np.random.choice(np.arange(n_nodes), num_refs, replace=False)
    ref_node_vals = node_vals[ref_node_idxs]
    ref_node_stddevs = 0.01 * np.ones(num_refs)
    with pytest.raises(AssertionError):
        infer_node_vals_and_errs(
            edge_idxs,
            obs_edge_diffs,
            edge_stddevs,
            ref_node_idxs,
            [],
            [],
        )
    infer_node_vals_and_errs(
        edge_idxs,
        obs_edge_diffs,
        edge_stddevs,
        ref_node_idxs,
        ref_node_vals,
        ref_node_stddevs,
    )
