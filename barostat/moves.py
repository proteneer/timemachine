import jax.numpy as jnp
import numpy as onp
from simtk import unit

from barostat.ensembles import NPTEnsemble
from barostat.utils import compute_box_volume, compute_box_center

from typing import List, Iterable, Tuple
from collections import namedtuple

CoordsAndBox = namedtuple('CoordsAndBox', ['coords', 'box'])


class CentroidRescaler:
    def __init__(self, group_inds, weights=None):
        self.group_inds = group_inds
        self.scatter_inds = self._scatter_inds_from_group_inds(group_inds)

        if weights is not None:
            raise(NotImplementedError)

    def _scatter_inds_from_group_inds(self, group_inds):
        """
        given a list of arrays of ints, representing groups of particles,
        construct a flat array, representing which group index to sort each particle into

        [[0,1,2], [3,4,5]] --> [0, 0, 0, 1, 1, 1]
        """
        all_inds = onp.hstack(group_inds)
        scatter_inds = onp.zeros(len(all_inds))

        # assert group_inds not overlapping
        assert (len(all_inds) == len(set(all_inds)))

        for i, group in enumerate(group_inds):
            for j in group:
                scatter_inds[j] = i

        return jnp.array(scatter_inds, dtype=int)

    def rescale(self, coords, center, scale=1.0):
        """scale distances of coords to center"""

        dx_initial = coords - center
        dx_updated = scale * dx_initial
        return center + dx_updated

    def compute_centroid(self, group):
        return jnp.mean(group, axis=0)

    def compute_centroids(self, coords):
        return jnp.array([self.compute_centroid(coords[inds]) for inds in self.group_inds])

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


class MonteCarloMove:
    n_proposed: int = 0
    n_accepted: int = 0

    def propose(self, x: CoordsAndBox) -> Tuple[CoordsAndBox, float]:
        """ return proposed state and log acceptance probability """
        raise NotImplementedError

    def move(self, x: CoordsAndBox) -> CoordsAndBox:
        proposal, log_acceptance_probability = self.propose(x)
        self.n_proposed += 1

        alpha = onp.random.rand()
        if alpha < onp.exp(log_acceptance_probability):
            self.n_accepted += 1
            return proposal
        else:
            return x


class MonteCarloBarostat(MonteCarloMove):
    def __init__(self,  # target_ensemble: NPTEnsemble,
                 reduced_potential_fxn: callable,
                 group_indices: List[Iterable[int]],
                 max_delta_volume: float = 0.05
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

        # how many molecule groups will we be adjusting...
        self.N = len(group_indices)

        self.centroid_rescaler = CentroidRescaler(group_indices)

    def propose(self, x: CoordsAndBox) -> Tuple[CoordsAndBox, float]:
        u_0 = self.reduced_potential_fxn(x.coords, x.box)
        volume = compute_box_volume(x.box)

        # sample uniformly in [-max_delta_volume, +max_delta_volume]
        delta_volume = (onp.random.rand() * 2 - 1) * self.max_delta_volume

        # apply scaling move
        # eq. 4 from Aqvist et al 2004
        proposed_volume = volume + delta_volume
        length_scale = (proposed_volume / volume) ** (1. / 3)

        proposed_coords = self.centroid_rescaler.scale_centroids(x.coords, compute_box_center(x.box), length_scale)

        proposed_box = length_scale * x.box

        proposed_state = CoordsAndBox(proposed_coords, proposed_box)

        u_proposed = self.reduced_potential_fxn(proposed_coords, proposed_box)
        delta_u = u_proposed - u_0

        jacobian_contribution = self.N * jnp.log(proposed_volume / volume)

        log_acceptance_probability = jnp.minimum(0, - (delta_u - jacobian_contribution))

        return proposed_state, log_acceptance_probability
