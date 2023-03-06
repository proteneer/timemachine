import networkx as nx
import numpy as np
from jax import jit
from jax import numpy as jnp
from jax import value_and_grad
from jax.scipy.stats import norm
from scipy.optimize import minimize


def make_stddevs_finite(stddevs, min_stddev=1e-3):
    """Ignore claims that stddev < min_stddev"""
    return jnp.maximum(stddevs, min_stddev)


def gaussian_log_likelihood(
    node_vals,  # [K] float array
    edge_idxs,  # [E, 2] int array, taking values in range(K)
    edge_diffs,  # [E] float array
    edge_stddevs,  # [E] float array
):
    r"""how likely are observed edge_diffs given underlying node_vals?

    likelihood(edge_diffs) = \prod_{i=0}^E
        Gaussian(
            edge_diffs[i];
            mean=node_vals[edge_idxs[i][1]] - node_vals[edge_idxs[i][0]],
            stddev=edge_stddevs[i],
        )

    References
    ----------
    [Xu, 2019] Optimal measurement network of pairwise differences
        https://doi.org/10.1021/acs.jcim.9b00528
        https://github.com/forcefield/DiffNet
    """
    src_idxs, dst_idxs = edge_idxs.T
    implied_diffs = node_vals[dst_idxs] - node_vals[src_idxs]

    # Note: can swap in other likelihood functions here besides norm.logpdf
    # (if we update the edge likelihood model, update bootstrapping as well)
    sanitized_edge_stddevs = make_stddevs_finite(edge_stddevs)
    edge_lls = norm.logpdf(x=edge_diffs, loc=implied_diffs, scale=sanitized_edge_stddevs)

    return jnp.sum(edge_lls)


def _assert_edges_complete(edge_idxs):
    K = np.max(edge_idxs) + 1
    assert set(range(K)) == set(edge_idxs.flatten())


def _assert_edges_connected(edge_idxs):
    g = nx.Graph(list(map(tuple, edge_idxs)))
    assert nx.number_connected_components(g) == 1


def _assert_edges_valid(edge_idxs):
    # some assertions for validity of a graph
    _assert_edges_complete(edge_idxs)
    _assert_edges_connected(edge_idxs)


def wrap_for_scipy_optimize(f):
    # utility for interfacing with scipy.optimize L-BFGS-B
    vg = jit(value_and_grad(f))

    def wrapped(x):
        v, g = vg(x)
        return float(v), np.array(g, dtype=np.float64)

    return wrapped


def infer_node_vals(edge_idxs, edge_diffs, edge_stddevs, ref_node_idxs=tuple(), ref_node_vals=tuple()):
    """
    Given pairwise comparisons involving K states,
    return a length-K vector of underlying absolute values
    by finding node_vals that maximizing likelihood of edge_diffs.

    Reference node vals influence result via a single additive offset.

    Parameters
    ----------
    edge_idxs: [E, 2] array, where the values are in range(K).
    edge_diffs: [E] array, relative values
    edge_stddevs: [E] array, positive

    ref_node_idxs: [N_ref] int array
    ref_node_vals: [N_ref] array

    Returns
    -------
        [K] array, inferred absolute values
    """

    # check shapes
    assert len(edge_diffs) == len(edge_idxs), f"{len(edge_diffs)} != {len(edge_idxs)}"
    _assert_edges_valid(edge_idxs)

    if len(ref_node_idxs) == 0:
        print("no reference node values: picking node 0 as arbitrary reference")
        ref_node_idxs = np.array([0], dtype=int)
        ref_node_vals = np.array([0], dtype=float)

    assert len(ref_node_idxs) == len(ref_node_vals), "Ref node idxs and ref node values must be of same length"

    # maximize likelihood of observed edge diffs, up to arbitrary offset
    @wrap_for_scipy_optimize
    def loss(x):
        # TODO: incorporate offset here?
        return -gaussian_log_likelihood(x, edge_idxs, edge_diffs, edge_stddevs)

    K = np.max(edge_idxs) + 1
    x0 = np.zeros(K)  # maybe initialize smarter, e.g. using random spanning tree?
    result = minimize(loss, x0, jac=True, tol=0).x

    centered_node_vals = result - result[0]

    # ref node vals only used to inform a single additive offset
    offset = np.mean(ref_node_vals - centered_node_vals[ref_node_idxs])

    return centered_node_vals + offset


def _bootstrap_node_vals(
    edge_idxs,
    edge_diffs,
    edge_stddevs,
    ref_node_idxs,
    ref_node_vals,
    ref_node_stddevs,
    n_bootstrap=100,
    seed=0,
):
    """call infer_node_vals multiple times with Gaussian bootstrapped edge_diffs, ref_node_vals"""

    n_edges = len(edge_idxs)
    n_nodes = len(set(edge_idxs.flatten()))
    n_refs = len(ref_node_idxs)

    rng = np.random.default_rng(seed)

    def estimate_node_val_w_offset(edge_vals, node_vals):
        return infer_node_vals(edge_idxs, edge_vals, edge_stddevs, ref_node_idxs, node_vals)

    bootstrap_estimates = np.zeros((n_bootstrap, n_nodes))

    for i in range(n_bootstrap):
        # if we switch the edge likelihood model, update this line too
        noisy_edge_diffs = edge_diffs + rng.standard_normal(n_edges) * edge_stddevs
        noisy_node_refs = ref_node_vals + rng.standard_normal(n_refs) * ref_node_stddevs

        bootstrap_estimates[i] = estimate_node_val_w_offset(noisy_edge_diffs, noisy_node_refs)

    return bootstrap_estimates


def infer_node_vals_and_errs(
    edge_idxs,
    edge_diffs,
    edge_stddevs,
    ref_node_idxs=tuple(),
    ref_node_vals=tuple(),
    ref_node_stddevs=tuple(),
    n_bootstrap=100,
    seed=0,
):
    """
    Given pairwise comparisons involving K states,
    return a length-K vector of underlying absolute values
    by finding node_vals that maximize likelihood of edge_diffs.

    Reference node vals influence result via a single additive offset.

    Parameters
    ----------
    edge_idxs: [E, 2] array, where the values are in range(K).
    edge_diffs: [E] array, relative values
    edge_stddevs: [E] array, positive

    ref_node_idxs: [N_ref] int array
    ref_node_vals: [N_ref] array
    ref_node_stddevs: [N_ref] array

    n_bootstrap: int
    seed: int

    Returns
    -------
    dg: [K] array
        inferred absolute values
    dg_err : [K] array
        associated errors (empirical stddev over bootstrap samples)
    """

    if len(ref_node_idxs) == 0:
        print("no reference node values: picking node 0 as arbitrary reference")
        ref_node_idxs = np.array([0], dtype=int)
        ref_node_vals = np.array([0], dtype=float)
        ref_node_stddevs = np.array([0], dtype=float)

    assert (
        len(ref_node_idxs) == len(ref_node_vals) == len(ref_node_stddevs)
    ), "Ref node idxs, ref node values and ref std devs must be of same length"

    dg = infer_node_vals(edge_idxs, edge_diffs, edge_stddevs, ref_node_idxs, ref_node_vals)

    # empirical stddev
    bootstrap_estimates = _bootstrap_node_vals(
        edge_idxs, edge_diffs, edge_stddevs, ref_node_idxs, ref_node_vals, ref_node_stddevs, n_bootstrap, seed
    )
    dg_err = bootstrap_estimates.std(0)

    return dg, dg_err
