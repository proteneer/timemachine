"""Summarize HarmonicBonds as Intervals in R+, construct maps between Intervals, & apply these maps to bond lengths"""

from dataclasses import dataclass

import numpy as np
from jax import config, jacobian, jit
from jax import numpy as jnp
from jax import vmap
from numpy.typing import NDArray

config.update("jax_enable_x64", True)

from typing import List

import networkx as nx

from timemachine.constants import BOLTZ, DEFAULT_TEMP


@dataclass
class Interval:
    lower: float
    upper: float

    @property
    def width(self):
        return self.upper - self.lower

    def validate(self):
        assert self.width > 0
        assert self.lower > 0


@dataclass
class Gaussian:
    mean: float
    stddev: float

    def to_interval(self, sigma_thresh=50):
        r = self.stddev * sigma_thresh
        interval = Interval(self.mean - r, self.mean + r)
        interval.validate()
        return interval


def gaussians_from_harmonic_bonds(ks, eq_lengths, temperature=DEFAULT_TEMP) -> List[Gaussian]:
    kBT = BOLTZ * temperature
    params = zip(ks, eq_lengths)
    return [Gaussian(eq_length, np.sqrt(kBT / k)) for (k, eq_length) in params]


def bond_length(x, y):
    # TODO: should this use periodic distance from timemachine.potentials.jax_utils?
    #   (I don't think that's necessary, since bonded atoms won't be in different periodic boxes...)
    return jnp.linalg.norm(x - y)


@jit
def interval_map(x, src_lb, src_ub, dst_lb, dst_ub):
    scale_factor = (dst_ub - dst_lb) / (src_ub - src_lb)

    in_support = (x >= src_lb) * (x <= src_ub)
    mapped = dst_lb + (x - src_lb) * scale_factor

    return jnp.where(in_support, mapped, np.nan)


def conf_map(x, bond, param):
    """Apply map to a single bond in a conformer x, return (updated x, logdetjac)"""
    a, b = bond
    dim = 3

    def f(xy, param):
        """R^[2 x dim] -> R^[2 x dim]"""
        x, y = xy[:dim], xy[dim:]
        src_lb, src_ub, dst_lb, dst_ub = param

        r = bond_length(x, y)
        new_r = interval_map(r, src_lb, src_ub, dst_lb, dst_ub)

        vec = (y - x) / jnp.linalg.norm(y - x)
        y_prime = x + new_r * vec

        xy_prime = jnp.hstack([x, y_prime])
        return xy_prime

    def map_and_logdetjac(x, y, param):
        xy = jnp.hstack([x, y])
        xy_prime = f(xy, param)
        y_prime = xy_prime[dim:]

        jac = jacobian(f)(xy, param)
        sign, logdet = jnp.linalg.slogdet(jac)

        # negative det would be unexpected --> nan-poison if detected
        logdetjac = jnp.where(sign == 1, logdet, jnp.nan)

        return y_prime, logdetjac

    x_b_mapped, logdetjac = map_and_logdetjac(x[a], x[b], param)
    x_updated = x.at[b].set(x_b_mapped)

    return x_updated, logdetjac


apply_conf_map_to_traj = jit(vmap(conf_map, in_axes=(0, None, None)))


def apply_conf_maps_to_traj(xs, bond_idxs, params):
    """Apply maps to several bonds in a conformer, return (updated x, logdetjac)"""
    xs_traj, logdetjac_increments_traj = [jnp.array(xs)], [np.zeros(len(xs))]

    for (bond, param) in zip(bond_idxs, params):  # TODO: jax.lax for-loop?
        xs_updated, logdetjac_increments = apply_conf_map_to_traj(xs_traj[-1], bond, param)

        xs_traj.append(xs_updated)
        logdetjac_increments_traj.append(logdetjac_increments)

    return xs_traj[-1], np.sum(logdetjac_increments_traj, axis=0)


# utilities for getting terminal bonds
def get_degrees(bond_idxs):
    g = nx.Graph()
    g.add_edges_from(bond_idxs)
    degrees = np.array([g.degree(i) for i in range(g.number_of_nodes())])
    return degrees


def get_terminal_bonds(bond_idxs):
    """Get bonded pairs that involve a terminal atom"""

    degrees = get_degrees(bond_idxs)
    sort_by_descending_degree = lambda bond: tuple(sorted(bond, key=lambda i: degrees[i], reverse=True))
    bonds = [sort_by_descending_degree(bond) for bond in bond_idxs]

    is_terminal = lambda bond: degrees[bond[1]] == 1

    terminal_bonds = sorted(filter(is_terminal, bonds))
    return terminal_bonds


class TerminalMappableState:
    def __init__(self, terminal_bond_idxs, ks, eq_lengths, temperature=DEFAULT_TEMP, sigma_thresh=20):
        """
        for each (a, b) in terminal_bond_idxs,
        prepare to construct a map that moves b -> b', conditioned on a

        terminal_bond_idxs, ks, eq_lengths:
            bonds that will be mapped,
            assumed in order (anchor, terminal) --> TODO: assert and/or automatically correct this

        temperature, sigma_thresh:
            determine the size of the interval
        """

        B = len(terminal_bond_idxs)
        assert (len(ks) == B) and (len(eq_lengths) == B)

        self.idxs = terminal_bond_idxs
        self.ks = ks
        self.eq_lengths = eq_lengths

        self.temperature = temperature
        self.sigma_thresh = sigma_thresh

        self.gaussians = gaussians_from_harmonic_bonds(ks, eq_lengths, DEFAULT_TEMP)
        self.intervals = [g.to_interval(sigma_thresh) for g in self.gaussians]

    def contains_in_support(self, x):
        """returns whether x is in the support of state"""

        bond_valid = []

        for i in range(len(self.idxs)):
            a, b = self.idxs[i]

            r = bond_length(x[a], x[b])
            interval = self.intervals[i]
            bond_valid.append((r <= interval.upper) * (r >= interval.lower))

        return jnp.array(bond_valid).all()

    @classmethod
    def from_harmonic_bond_params(cls, bond_idxs, params, temperature=DEFAULT_TEMP, sigma_thresh=20):
        # bond (i, j) -> (k, eq_length)
        param_dict = dict(zip(map(tuple, bond_idxs), params))  # bond idxs may be in arbitrary order

        terminal_bond_tuples = get_terminal_bonds(bond_idxs)  # each tuple will be sorted in order (anchor, terminal)

        ks, eq_lengths = np.array([param_dict[tuple(sorted(bond))] for bond in terminal_bond_tuples]).T

        return cls(np.array(terminal_bond_tuples), ks, eq_lengths, temperature=temperature, sigma_thresh=sigma_thresh)


def states_to_conf_map_params(src: TerminalMappableState, dst: TerminalMappableState):
    # find bond idxs in common
    src_bonds = set(tuple(b) for b in src.idxs)
    dst_bonds = set(tuple(b) for b in dst.idxs)
    bonds_in_common = src_bonds.intersection(dst_bonds)

    bond_idxs = np.array(list(bonds_in_common))
    assert len(bond_idxs) > 0 and bond_idxs.shape[1] == 2 and len(bond_idxs.shape) == 2

    params_list = []

    for (a, b) in bond_idxs:
        src_interval = [interval for (idx, interval) in zip(src.idxs, src.intervals) if tuple(idx) == (a, b)][0]
        dst_interval = [interval for (idx, interval) in zip(dst.idxs, dst.intervals) if tuple(idx) == (a, b)][0]

        params_list.append((src_interval.lower, src_interval.upper, dst_interval.lower, dst_interval.upper))

    params = np.array(params_list)

    return bond_idxs, params


@dataclass()
class TerminalBondMap:
    mapped_bond_idxs: NDArray
    map_params: NDArray

    @classmethod
    def from_states(cls, src: TerminalMappableState, dst: TerminalMappableState):
        bond_idxs, params = states_to_conf_map_params(src, dst)
        return cls(bond_idxs, params)

    def __call__(self, xs):
        return apply_conf_maps_to_traj(xs, self.mapped_bond_idxs, self.map_params)
