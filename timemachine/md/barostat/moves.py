import numpy as np
from jax import numpy as jnp
from jax.ops import segment_sum
from numpy.typing import NDArray

from timemachine import lib
from timemachine.lib import custom_ops
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.moves import NVTMove
from timemachine.md.states import CoordsVelBox
from timemachine.potentials import BoundPotential, HarmonicBond
from timemachine.potentials.potential import get_bound_potential_by_type


def compute_centroid(group):
    return jnp.mean(group, axis=0)


def _scatter_inds_from_group_inds(group_inds):
    """
    given a list of arrays of ints, representing groups of particles,
    construct a flat array, representing which group index to sort each particle into

    [[0,1,2], [3,4,5]] --> [0, 0, 0, 1, 1, 1]
    """
    all_inds = np.hstack(group_inds)
    scatter_inds = np.zeros(len(all_inds))

    # assert group_inds not overlapping
    assert len(all_inds) == len(set(all_inds))

    for i, group in enumerate(group_inds):
        for j in group:
            scatter_inds[j] = i

    return np.array(scatter_inds, dtype=int)


class CentroidRescaler:
    def __init__(self, group_inds, weights=None):
        self.group_inds = group_inds
        self.group_sizes = jnp.array(list(map(len, self.group_inds)))
        assert jnp.min(self.group_sizes) > 0

        self.scatter_inds = _scatter_inds_from_group_inds(group_inds)

        if weights is not None:
            raise NotImplementedError("Weights are not implemented yet")

    def rescale(self, coords, center, scale=1.0):
        """scale distances of coords to center"""

        dx_initial = coords - center
        dx_updated = scale * dx_initial
        return center + dx_updated

    def compute_centroids(self, coords):
        """Returns an array containing the centroids of each group"""
        return segment_sum(coords, self.scatter_inds) / jnp.expand_dims(self.group_sizes, axis=1)

    def _slow_compute_centroids(self, coords):
        """For testing / reference"""
        return jnp.array([compute_centroid(coords[g]) for g in self.group_inds])

    def displace_by_group(self, coords, displacements):
        return coords + displacements[self.scatter_inds]

    def scale_centroids(self, coords, center, scale):
        """

        Notes
        -----
        * Currently ignores particle mass -- the centroid of a group of
            particles will be computed assuming all particles have equal weight
        * Later, particle weights could be set in some arbitrary way within each group
        """

        centroids = self.compute_centroids(coords)
        group_displacements = self.rescale(centroids, center, scale) - centroids
        displaced_coords = self.displace_by_group(coords, group_displacements)

        return displaced_coords


class NPTMove(NVTMove):
    """
    Functionally, NPT is implemented as NVTMove plus a MC Barostat.
    So inherit from NVTMove here.
    """

    def __init__(
        self,
        bps: list[BoundPotential],
        masses: NDArray,
        temperature: float,
        pressure: float,
        n_steps: int,
        seed: int,
        dt: float = 1.5e-3,
        friction: float = 1.0,
        barostat_interval: int = 5,
    ):
        super().__init__(bps, masses, temperature, n_steps, seed, dt=dt, friction=friction)

        bonded_pot = get_bound_potential_by_type(bps, HarmonicBond).potential

        bond_list = get_bond_list(bonded_pot)
        group_idxs = get_group_indices(bond_list, len(masses))

        barostat = lib.MonteCarloBarostat(len(masses), pressure, temperature, group_idxs, barostat_interval, seed + 1)
        barostat_impl = barostat.impl(self.bound_impls)
        self.barostat_impl = barostat_impl

    def move(self, x: CoordsVelBox) -> CoordsVelBox:
        # note: context creation overhead here is actually very small!
        ctxt = custom_ops.Context(
            x.coords, x.velocities, x.box, self.integrator_impl, self.bound_impls, movers=[self.barostat_impl]
        )
        return self._steps(ctxt)
