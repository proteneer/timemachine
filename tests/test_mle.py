import hashlib
import random
from functools import partial

import networkx as nx
import numpy as np
import pytest
from scipy.stats import linregress

from timemachine.fe.mle import infer_node_vals, infer_node_vals_and_errs, infer_node_vals_and_errs_networkx

pytestmark = [pytest.mark.nocuda]


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
            n_bootstrap=10,
            seed=np.random.randint(1000),
        )
        assert (dg_err > 0).all()

        res = linregress(dg, node_vals)
        assert res.rvalue > 0.9


def test_infer_node_dgs_w_error_invariant_wrt_edge_order():
    "Check that permuting the edges doesn't affect the result significantly"
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

        seed = np.random.randint(1000)

        f = partial(
            infer_node_vals_and_errs,
            ref_node_idxs=ref_node_idxs,
            ref_node_vals=ref_node_vals,
            ref_node_stddevs=ref_node_stddevs,
            n_bootstrap=10,
            seed=seed,
        )

        dg_1, dg_err_1 = f(edge_idxs, obs_edge_diffs, edge_stddevs)

        p = np.random.permutation(len(edge_idxs))
        dg_2, dg_err_2 = f(edge_idxs[p, :], obs_edge_diffs[p], edge_stddevs[p])

        np.testing.assert_allclose(dg_1, dg_2, rtol=1e-4)

        # TODO: errors are noisy; unclear how to test consistency
        # np.testing.assert_allclose(dg_err_1, dg_err_2) # fails


edge_diff_prop = "edge_diff"
edge_stddev_prop = "edge_stddev"
node_val_prop = "node_val"
node_stddev_prop = "node_stddev"
ref_node_val_prop = "ref_node_val"
ref_node_stddev_prop = "ref_node_stddev"
n_bootstrap = 10


@pytest.fixture(scope="module", params=[0])
def _nx_graph_with_reference_mle_instance(request):
    seed = request.param
    np.random.seed(seed)

    edge_noise_stddev = np.random.rand()
    g = generate_random_valid_regular_graph()
    g = nx.convert_node_labels_to_integers(g)
    n_nodes = g.number_of_nodes()

    node_vals, edge_idxs, obs_edge_diffs, edge_stddevs = generate_instance(g, edge_noise_stddev)

    num_refs = np.random.randint(n_nodes)
    ref_node_idxs = np.random.choice(np.arange(n_nodes), num_refs, replace=False)
    ref_node_vals = node_vals[ref_node_idxs]
    ref_node_stddevs = 0.01 * np.ones(num_refs)

    g = nx.DiGraph()

    g.add_nodes_from(range(n_nodes))

    for (u, v), diff, stddev in zip(edge_idxs, obs_edge_diffs, edge_stddevs):
        g.add_edge(u, v, **{edge_diff_prop: diff, edge_stddev_prop: stddev})

    for n, ref_val, ref_stddev in zip(ref_node_idxs, ref_node_vals, ref_node_stddevs):
        g.add_node(n, **{ref_node_val_prop: ref_val, ref_node_stddev_prop: ref_stddev})

    dgs, dg_errs = infer_node_vals_and_errs(
        edge_idxs, obs_edge_diffs, edge_stddevs, ref_node_idxs, ref_node_vals, ref_node_stddevs, n_bootstrap, seed
    )

    return g, seed, dgs, dg_errs


@pytest.fixture(scope="function")
def nx_graph_with_reference_mle_instance(_nx_graph_with_reference_mle_instance):
    g, *xs = _nx_graph_with_reference_mle_instance
    return (g.copy(), *xs)


infer_node_vals_and_errs_networkx_partial = partial(
    infer_node_vals_and_errs_networkx,
    edge_diff_prop=edge_diff_prop,
    edge_stddev_prop=edge_stddev_prop,
    ref_node_val_prop=ref_node_val_prop,
    ref_node_stddev_prop=ref_node_stddev_prop,
    node_val_prop=node_val_prop,
    node_stddev_prop=node_stddev_prop,
    n_bootstrap=n_bootstrap,
)


def test_infer_node_vals_and_errs_networkx_requires_digraph():
    g = nx.Graph()
    with pytest.raises(AssertionError, match="Graph must be a DiGraph"):
        infer_node_vals_and_errs_networkx_partial(g)


def test_infer_node_vals_and_errs_networkx(nx_graph_with_reference_mle_instance):
    g, seed, ref_dgs, ref_dg_errs = nx_graph_with_reference_mle_instance

    g_res = infer_node_vals_and_errs_networkx_partial(g, seed=seed)

    for n, (ref_dg, ref_dg_err) in enumerate(zip(ref_dgs, ref_dg_errs)):
        assert g_res.nodes[n][node_val_prop] == pytest.approx(ref_dg, rel=1e-5)
        # TODO: errors are noisy; unclear how to test consistency
        # assert g_res.nodes[n][node_stddev_prop] == pytest.approx(ref_dg_err)


def test_infer_node_vals_and_errs_networkx_multi(nx_graph_with_reference_mle_instance):
    g, seed, ref_dgs, ref_dg_errs = nx_graph_with_reference_mle_instance

    # MultiDiGraph should result in same values
    mg = nx.MultiDiGraph()
    for n, d in g.nodes.items():
        mg.add_node(n, **d)
    for e, d in g.edges.items():
        mg.add_edge(*e, **d)

    g_res = infer_node_vals_and_errs_networkx_partial(mg, seed=seed)
    for n, (ref_dg, ref_dg_err) in enumerate(zip(ref_dgs, ref_dg_errs)):
        assert g_res.nodes[n][node_val_prop] == pytest.approx(ref_dg, rel=1e-5)
        assert g_res.nodes[n][node_stddev_prop] == pytest.approx(ref_dg_err, rel=1e-5)

    # multiple edges should result in similar values
    for e, d in g.edges.items():
        assert mg.add_edge(*e, **d) == 1

    g_res = infer_node_vals_and_errs_networkx_partial(mg, seed=seed)
    for n, (ref_dg, ref_dg_err) in enumerate(zip(ref_dgs, ref_dg_errs)):
        assert g_res.nodes[n][node_val_prop] == pytest.approx(ref_dg, abs=1e-5)

    # adding all 0 edges should shift the dgs toward 0
    for e, d in g.edges.items():
        assert mg.add_edge(*e, **{edge_diff_prop: 0.0, edge_stddev_prop: 0.1}) == 2

    g_res = infer_node_vals_and_errs_networkx_partial(mg, seed=seed)
    for n, (ref_dg, ref_dg_err) in enumerate(zip(ref_dgs, ref_dg_errs)):
        assert np.abs(g_res.nodes[n][node_val_prop]) < 1.0


def test_infer_node_vals_and_errs_networkx_invariant_wrt_permutation(nx_graph_with_reference_mle_instance):
    "Ensure result does not depend on the internal ordering of nodes and edges in the networkx graph"

    g, seed, ref_dgs, ref_dg_errs = nx_graph_with_reference_mle_instance

    g_shuffled = nx.DiGraph()
    nodes = list(g.nodes.items())
    edges = list(g.edges.items())
    random.seed(0)
    random.shuffle(nodes)
    random.shuffle(edges)
    g_shuffled.add_nodes_from(nodes)
    g_shuffled.add_edges_from((u, v, d) for (u, v), d in edges)

    # use n_bootstrap=2 to save time, since we don't check errors here
    g_res = infer_node_vals_and_errs_networkx_partial(g_shuffled, n_bootstrap=2, seed=seed)

    for n, (ref_dg, ref_dg_err) in enumerate(zip(ref_dgs, ref_dg_errs)):
        assert g_res.nodes[n][node_val_prop] == pytest.approx(ref_dg, rel=1e-4)
        # TODO: errors are noisy; unclear how to test consistency
        # assert g_res.nodes[n][node_stddev_prop] == pytest.approx(ref_dg_err)


def test_infer_node_vals_and_errs_networkx_invariant_wrt_relabeling_nodes(nx_graph_with_reference_mle_instance):
    "Ensure results are invariant wrt relabeling nodes"

    g, seed, ref_dgs, ref_dg_errs = nx_graph_with_reference_mle_instance

    idx_to_label = {n: hashlib.sha256(bytes(n)).hexdigest() for n in g.nodes}
    g_relabeled = nx.relabel_nodes(g, idx_to_label)

    # use n_bootstrap=2 to save time, since we don't check errors here
    g_res = infer_node_vals_and_errs_networkx_partial(g_relabeled, n_bootstrap=2, seed=seed)

    for n, (ref_dg, ref_dg_err) in enumerate(zip(ref_dgs, ref_dg_errs)):
        assert g_res.nodes[idx_to_label[n]][node_val_prop] == pytest.approx(ref_dg, rel=1e-5)
        # TODO: errors are noisy; unclear how to test consistency
        # assert g_relabeled_res.nodes[idx_to_label[n]][node_stddev_prop] == pytest.approx(ref_dg_err)


def test_infer_node_vals_and_errs_networkx_missing_values(nx_graph_with_reference_mle_instance):
    "Check that edges with missing values are ignored"

    np.random.seed(0)

    g, seed, ref_dgs, ref_dg_errs = nx_graph_with_reference_mle_instance

    idx_to_label = {n: str(n) for n in g.nodes}
    g = nx.relabel_nodes(g, idx_to_label)
    n1, n2, n3 = np.random.choice(g.nodes, 3, replace=False)

    # define a new node somewhere in the middle of the sorted list of nodes
    # (needed to check that we correctly remove isolated nodes)
    undetermined_label = str(np.random.randint(1, g.number_of_nodes() - 1)) + "_undetermined"
    g.add_edge(n1, undetermined_label)
    g.add_edge(n2, undetermined_label, **{edge_diff_prop: None})
    g.add_edge(n3, undetermined_label, **{edge_diff_prop: None, edge_stddev_prop: None})

    # unlabeled edges between exising nodes should have no effect on result
    g.add_edge(n1, n2)
    g.add_edge(n2, n3, **{edge_diff_prop: None})
    g.add_edge(n1, n3, **{edge_diff_prop: None, edge_stddev_prop: None})

    # use n_bootstrap=2 to save time, since we don't check errors here
    g_res = infer_node_vals_and_errs_networkx_partial(g, n_bootstrap=2, seed=seed)

    for n, (ref_dg, ref_dg_err) in enumerate(zip(ref_dgs, ref_dg_errs)):
        assert g_res.nodes[idx_to_label[n]][node_val_prop] == pytest.approx(ref_dg, rel=1e-5)
        # TODO: errors are noisy; unclear how to test consistency
        # assert g_res.nodes[idx_to_label[n]][node_stddev_prop] == ref_dg_err

    # undetermined node should not be in the result
    assert undetermined_label not in g_res.nodes


def test_infer_node_vals_and_errs_networkx_raises_on_empty():
    g = nx.DiGraph()
    with pytest.raises(ValueError, match="Empty graph"):
        infer_node_vals_and_errs_networkx_partial(g)
    g.add_edge(0, 1)
    with pytest.raises(ValueError, match="Empty graph after removing edges without predictions"):
        infer_node_vals_and_errs_networkx_partial(g)


def test_infer_node_vals_incorrect_sizes():
    """Verify that infer_node_vals correctly asserts that the length of arrays are the same"""
    np.random.seed(2023)
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
    np.random.seed(2023)
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


def instance_to_nx(instance):
    node_vals, edge_idxs, obs_edge_diffs, edge_stddevs = instance
    g = nx.DiGraph()
    for i in range(len(edge_idxs)):
        e = tuple(edge_idxs[i])
        d = dict()
        d["edge_diff"] = obs_edge_diffs[i]
        d["edge_stddev"] = edge_stddevs[i]
        g.add_edge(*e, **d)
    return g


def compare_inferred_and_ref_dgs(inferred_dgs: dict, node_vals, mse_thresh=1e-5):
    nodes = sorted(inferred_dgs.keys())
    ref_node = nodes[0]
    x = []
    y = []
    for n in inferred_dgs:
        x.append(inferred_dgs[n] - inferred_dgs[ref_node])
        y.append(node_vals[n] - node_vals[ref_node])
    mse = np.mean((np.array(y) - np.array(x)) ** 2)
    assert mse < mse_thresh, f"{mse} >= {mse_thresh}"


def test_disconnection():
    """infer dgs on largest connected component of disconnected random trees"""
    np.random.seed(0)
    for i in range(5):
        # generate random spanning tree, remove one random edge
        K = np.random.randint(10, 100)
        g = nx.random_tree(K, seed=hash(K * i))
        edges = list(g.edges)
        random_edge = edges[np.random.randint(len(edges))]
        g.remove_edge(*random_edge)
        size_of_largest_component = max([len(c) for c in nx.connected_components(g)])
        assert size_of_largest_component < K

        # convert to digraph with appropriate edge labels
        instance = generate_instance(g, 1e-3)
        test_g = instance_to_nx(instance)
        node_vals, edge_idxs, obs_edge_diffs, edge_stddevs = instance

        # infer results
        labeled_graph = infer_node_vals_and_errs_networkx(
            test_g, "edge_diff", "edge_stddev", "ref", "ref_stddev", n_bootstrap=1
        )
        assert labeled_graph.number_of_nodes() == size_of_largest_component

        # assert inferred dgs match ref dgs, up to an additive offset
        inferred_dgs = nx.get_node_attributes(labeled_graph, "inferred_dg")
        assert len(inferred_dgs) == size_of_largest_component
        compare_inferred_and_ref_dgs(inferred_dgs, node_vals)
