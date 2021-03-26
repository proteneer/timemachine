import jax.numpy as jnp
import numpy as onp
from simtk import unit

from barostat.ensembles import NPTEnsemble
from barostat.utils import compute_box_volume, compute_box_center

from typing import List, Iterable, Tuple
from collections import namedtuple

CoordsAndBox = namedtuple('CoordsAndBox', ['coords', 'box'])


def rescale(coords, center, scale=1.0):
    """scale distances of coords to center"""

    dx_initial = coords - center
    dx_updated = scale * dx_initial
    return center + dx_updated


def compute_centroid(group, weights=None):
    if weights is not None:
        assert (not (weights < 0).any())
        normalized_weights = weights / jnp.sum(weights)
        return jnp.mean(normalized_weights * group, axis=0)

    return jnp.mean(group, axis=0)


def compute_centroids(coords, group_inds, weights=None):
    return jnp.array([compute_centroid(coords[inds], weights) for inds in group_inds])


def displace_by_group(coords, group_inds, displacements):
    # note: this implementation will fail silently for overlapping group_inds
    # TODO: assert group_inds not overlapping

    displaced_coords = jnp.array(coords)
    for (inds, displacement) in zip(group_inds, displacements):
        displaced_coords[inds] += displacement

    return displaced_coords


def scale_centroids(coords, center, group_inds, scale, weights=None):
    """

    Notes
    -----
    * Currently ignores particle mass -- the centroid of a group of
        particles will be computed assuming all particles have equal weight
    * Later, particle weights could be set in some arbitrary way within each group
    """

    centroids = compute_centroids(coords, group_inds, weights)
    group_displacements = rescale(centroids, center, scale) - centroids
    displaced_coords = displace_by_group(coords, group_inds, group_displacements)

    return displaced_coords


# TODO: group the above few functions into a CentroidScaler class


class MonteCarloMove:
    def propose(self, x: CoordsAndBox) -> Tuple[CoordsAndBox, float]:
        """ return proposed state and log acceptance probability """
        raise NotImplementedError

    def move(self, x: CoordsAndBox) -> CoordsAndBox:
        proposal, log_acceptance_probability = self.propose(x)
        alpha = onp.random.rand()
        if alpha < onp.exp(log_acceptance_probability):
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

    def propose(self, x: CoordsAndBox) -> Tuple[CoordsAndBox, float]:
        u_0 = self.reduced_potential_fxn(x.coords, x.box)
        volume = compute_box_volume(x.box)

        # sample uniformly in [-max_delta_volume, +max_delta_volume]
        delta_volume = (onp.random.rand() * 2 - 1) * self.max_delta_volume

        # apply scaling move
        # eq. 4 from Aqvist et al 2004
        proposed_volume = volume + delta_volume
        length_scale = (proposed_volume / volume) ** (1. / 3)

        proposed_coords = scale_centroids(x.coords, compute_box_center(x.box), group_inds=self.group_indices,
                                          scale=length_scale)
        proposed_box = length_scale * x.box

        proposed_state = CoordsAndBox(proposed_coords, proposed_box)

        u_proposed = self.reduced_potential_fxn(proposed_coords, proposed_box)
        delta_u = u_proposed - u_0

        jacobian_contribution = self.N * jnp.log(proposed_volume / volume)

        log_acceptance_probability = jnp.maximum(0, - (delta_u - jacobian_contribution))

        return proposed_state, log_acceptance_probability
