# alternatives to decoupling all atoms at once
# adapted from:
#   https://github.com/proteneer/timemachine/blob/acccb46c3ed7eaf614fd79b42e42db597a546891/examples/sifting.py#L212-L328

import networkx as nx
import numpy as np
from jax import numpy as jnp

from timemachine.graph_utils import convert_to_nx


def rank_atoms_by_path_length_to_src(mol, source_idxs):
    """argsort by rank(i) = min_{s in source_idxs} shortest_path_length(i, s)

    Parameters
    ----------
    mol : rdkit mol
    source_idxs : int sequence

    Returns
    -------
    ranks : int array
        a permutation of range(mol.GetNumAtoms())
        sorted by distance from source_idxs
        (arbitrarily breaks ties)

    Usage suggestion
    ----------------
    * source_idxs might contain a single central atom, a list of anchor atoms, or a list of peripheral atoms
    """

    graph = convert_to_nx(mol)
    _distance_dict = nx.multi_source_dijkstra_path_length(graph, sources=list(source_idxs))
    min_dist = np.array([_distance_dict[i] for i in range(mol.GetNumAtoms())])

    if not (min_dist < np.inf).all():
        raise ValueError("didn't expect mol graph to be disconnected")

    ranks = np.argsort(min_dist)
    return ranks


# TODO: add utilities to get atom ranks using other heuristics,
#   e.g. by distance from periphery, or by distance from center, or looking at conformer rather than only graph?


class DecoupleByAtomRank:
    def __init__(self, atom_idxs, atom_ranks):
        """Interpolate atom_lams[i] from 0 to 1 in stages, with stages defined by atom_ranks.

        Notes
        -----
        this function doesn't attempt to handle
            (1) indexing properly into a system's collection of nb_params,
            (2) converting atom lams into w_offsets or other parameters.

        TODO: modify to accept an "overlap" parameter?
            when "overlap" == 0, stage i completes before stage i+1 begins (as in current implementation)
            when "overlap" == 1, all stages run at once (aka atom_lams[i] = global_lam for all i)
            and get maybe useful intermediate behaviors for 0 < overlap < 1
        """
        self.atom_idxs = atom_idxs
        self.atom_ranks = atom_ranks
        self.num_stages = len(set(atom_ranks))
        assert set(atom_ranks) == set(range(self.num_stages))

    def atom_lams_from_global_lam(self, global_lam):
        """each stage goes from 0 to 1 in turn"""

        num_atoms = len(self.atom_idxs)

        bin_width = 1 / self.num_stages
        fractional_ranks = (self.atom_ranks + 1) / self.num_stages

        upper_boundaries = fractional_ranks
        lower_boundaries = upper_boundaries - bin_width

        atom_lams = jnp.clip(self.num_stages * (global_lam - lower_boundaries), 0, 1)

        # patch special case: ensure exactly 1.0 at endpoint
        # (since in jax default float32 mode, atom_lams can be 0.99999994 != 1.0 at global_lam = 1.0)
        _ones = jnp.ones(num_atoms)
        atom_lams = jnp.where(global_lam == _ones, _ones, atom_lams)

        return atom_lams
