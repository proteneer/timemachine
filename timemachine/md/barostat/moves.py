from jax import config
from jax import numpy as jnp
from jax.ops import segment_sum

config.update("jax_enable_x64", True)

from typing import Iterable, List, Tuple

import numpy as onp

from timemachine.md.barostat.utils import compute_box_center, compute_box_volume
from timemachine.md.moves import MonteCarloMove
from timemachine.md.states import CoordsVelBox


def compute_centroid(group):
    return jnp.mean(group, axis=0)


def _scatter_inds_from_group_inds(group_inds):
    """
    given a list of arrays of ints, representing groups of particles,
    construct a flat array, representing which group index to sort each particle into

    [[0,1,2], [3,4,5]] --> [0, 0, 0, 1, 1, 1]
    """
    all_inds = onp.hstack(group_inds)
    scatter_inds = onp.zeros(len(all_inds))

    # assert group_inds not overlapping
    assert len(all_inds) == len(set(all_inds))

    for i, group in enumerate(group_inds):
        for j in group:
            scatter_inds[j] = i

    return jnp.array(scatter_inds, dtype=int)


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


class MonteCarloBarostat(MonteCarloMove):
    def __init__(
        self,  # target_ensemble: NPTEnsemble,
        reduced_potential_fxn: callable,
        group_indices: List[Iterable[int]],
        max_delta_volume: float = 0.05,
        adapt_proposal_scale: bool = True,
    ):
        """

        References
        ----------
        * OpenMM MonteCarloBarostat implementation + theory documentation
            * https://github.com/openmm/openmm/blob/be19e0222ddf66f612016a3c1f687161a53c2396/openmmapi/src/MonteCarloBarostatImpl.cpp#L64
            * http://docs.openmm.org/latest/userguide/theory.html#montecarlobarostat
        * Kim-Hung Chow and David M. Ferguson.
            Isothermal-isobaric molecular dynamics simulations with Monte Carlo volume sampling.
            Computer Physics Communications, 91:283–289, 1995.
        * Johan Åqvist, Petra Wennerström, Martin Nervall, Sinisa Bjelic, and Bjørn O. Brandsdal.
            Molecular dynamics simulations of water and biomolecules with a Monte Carlo constant pressure algorithm.
            Chemical Physics Letters, 384:288–294, 2004.
        """
        # TODO: make up my mind whether this constructor should accept an NPTEnsemble or a reduced_potential_fxn
        # self.target_ensemble = target_ensemble
        self.reduced_potential_fxn = reduced_potential_fxn

        self.group_indices = group_indices
        self.max_delta_volume = max_delta_volume
        self._initial_max_delta_volume = max_delta_volume

        # how many molecule groups will we be adjusting...
        self.N = len(group_indices)

        self.centroid_rescaler = CentroidRescaler(group_indices)
        self.adapt_proposal_scale = adapt_proposal_scale

    def propose(self, x: CoordsVelBox) -> Tuple[CoordsVelBox, float]:
        u_0 = self.reduced_potential_fxn(x.coords, x.box)
        volume = compute_box_volume(x.box)

        # sample uniformly in [-max_delta_volume, +max_delta_volume]
        delta_volume = (onp.random.rand() * 2 - 1) * self.max_delta_volume

        # apply scaling move
        # eq. 4 from Aqvist et al 2004
        proposed_volume = volume + delta_volume
        length_scale = (proposed_volume / volume) ** (1.0 / 3)

        proposed_coords = self.centroid_rescaler.scale_centroids(x.coords, compute_box_center(x.box), length_scale)

        proposed_box = length_scale * x.box

        proposed_state = CoordsVelBox(proposed_coords, x.velocities, proposed_box)

        u_proposed = self.reduced_potential_fxn(proposed_coords, proposed_box)
        delta_u = u_proposed - u_0

        jacobian_contribution = self.N * jnp.log(proposed_volume / volume)

        log_acceptance_probability = jnp.minimum(0, -(delta_u - jacobian_contribution))

        return proposed_state, log_acceptance_probability

    def adapt(self):
        """Adapt self.max_delta_volume if the recent acceptance fraction is outside of range [0.25, 0.75]

        Notes
        -----
        TODO: refactor so that this doesn't overwrite the global barostat.n_proposed, barostat.n_accepted counters,
            instead using something like barostat.n_proposed_since_last_adaptation?

        References
        ----------
        Direct clone of https://github.com/openmm/openmm/blob/d8ef57fed6554ec95684e53768188e1f666405c9/openmmapi/src/MonteCarloBarostatImpl.cpp#L103-L113
        with the following tweaks:
            * introduces a lower_bound (so that the proposal scale can't go below some preset value)
            * introduces an upper_bound (so that the proposal scale can't go above some preset value,
                rather than using the state-dependent limit 0.3 * current_box_volume in OpenMM)

        """
        adaptation_multiplier = 1.1  # multiply/divide by this factor when increasing/decreasing proposal scale
        lower_bound = 1e-3  # don't let self.max_delta_volume drop below lower_bound nm^3
        upper_bound = 1e1  # don't let self.max_delta_volume exceed upper_bound nm^3

        if self.n_proposed >= 10:

            if self.acceptance_fraction < 0.25:
                decreased = self.max_delta_volume / adaptation_multiplier
                self.max_delta_volume = max(lower_bound, decreased)
                self.reset_counters()

            elif self.acceptance_fraction > 0.75:
                increased = self.max_delta_volume * adaptation_multiplier
                self.max_delta_volume = min(upper_bound, increased)
                self.reset_counters()

    def reset_counters(self):
        self.n_proposed = 0
        self.n_accepted = 0

    def reset_proposal_scale(self):
        self.max_delta_volume = self._initial_max_delta_volume

    def reset(self):
        self.reset_counters()
        self.reset_proposal_scale()

    def move(self, x: CoordsVelBox) -> CoordsVelBox:
        x_next = super().move(x)
        if self.adapt_proposal_scale:
            self.adapt()
        return x_next
