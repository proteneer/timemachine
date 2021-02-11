import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS
from fe.topology import AtomMappingError


class CompareDist(rdFMCS.MCSAtomCompare):
    """Custom atom comparison: use positions within generated conformer"""

    def __init__(self, threshold=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def compare(self, p, mol1, atom1, mol2, atom2):
        """Atoms match if within 0.5 Å

        Signature from super method:
        (MCSAtomCompareParameters)parameters, (Mol)mol1, (int)atom1, (Mol)mol2, (int)atom2) -> bool
        """
        x_i = mol1.GetConformer(0).GetPositions()[atom1]
        x_j = mol2.GetConformer(0).GetPositions()[atom2]
        return bool(np.linalg.norm(x_i - x_j) <= self.threshold)  # must convert from np.bool_ to Python bool!


def mcs_map(a, b, threshold=0.5):
    """Find the MCS map of going from A to B"""
    params = rdFMCS.MCSParameters()
    params.AtomTyper = CompareDist(threshold=threshold)
    return rdFMCS.FindMCS([a, b], params)


def transformation_size(n_A, n_B, n_MCS):
    """Heuristic size of transformation in terms of
	the number of atoms in A, B, and MCS(A, B)

    Notes
    -----
    * YTZ suggests (2021/01/26)
        (n_A + n_B) - n_MCS
	which is size-extensive

    * JF modified (2021/01/27)
        (n_A - n_MCS) + (n_B - n_MCS)
	which is 0 when A = B

    * Alternatives considered but not tried:
        * min(n_A, n_B) - n_MCS
        * max(n_A, n_B) - n_MCS
        * 1.0 - (n_MCS / min(n_A, n_B))

    TODO: move this into a utility module or the free energy module
    """
    return (n_A + n_B) - 2 * n_MCS


def compute_all_pairs_mcs(mols):
    """Generate square matrix of MCS(mols[i], mols[j]) for all pairs i,j

    Parameters
    ----------
    mols : iterable of RDKit Mols

    Returns
    -------
    mcs_s : numpy.ndarray of rdkit.Chem.rdFMCS.MCSResult's, of shape (len(mols), len(mols))

    Notes
    -----
    TODO: mcs_map should be symmetric, so only have to do the upper triangle of this matrix
    """
    mcs_s = np.zeros((len(mols), len(mols)), dtype=object)

    for i in range(len(mols)):
        for j in range(len(mols)):
            mcs_s[i, j] = mcs_map(mols[i], mols[j])
    return mcs_s


def compute_transformation_size_matrix(mols, mcs_s):
    N = len(mols)
    transformation_sizes = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            n_i = mols[i].GetNumAtoms()
            n_j = mols[j].GetNumAtoms()
            n_mcs = mcs_s[i, j].numAtoms
            transformation_sizes[i, j] = transformation_size(n_i, n_j, n_mcs)
    return transformation_sizes


# 4. Construct and serialize the relative transformations
def _check_core_map_distances(mol_a, mol_b, core, threshold=0.5) -> bool:
    """compute vector of distances[i] = distance(conf_a[core_a[i]], conf_b[core_b[i]]),
    check whether distances[i] <= threshold for all i"""

    a, b = core[:, 0], core[:, 1]
    conf_a = mol_a.GetConformer(0).GetPositions()
    conf_b = mol_b.GetConformer(0).GetPositions()
    distances = np.linalg.norm(conf_a[a] - conf_b[b], axis=1)
    return (distances <= threshold).all()


def get_core_by_mcs(mol_a, mol_b, query, threshold=0.5):
    """Return np integer array that can be passed to RelativeFreeEnergy constructor

    Parameters
    ----------
    mol_a, mol_b, query : RDKit molecules
    threshold : float, in angstroms

    Returns
    -------
    core : np.ndarray of ints, shape (n_MCS, 2)

    Notes
    -----
    * Warning! Some atoms that intuitively should be mapped together are not,
        when threshold=0.5 Å in custom atom comparison, because conformers aren't
        quite aligned enough.
    * Warning! Because of the intermediate representation of a substructure query,
        the core indices can get flipped around,
        for example if the substructure match hits only part of an aromatic ring.

        In some cases, this means that pairs of atoms that do not satisfy the
        atom comparison function can be mapped together.

    TODO: move this into a utility module or the free energy module
    """

    # fetch conformer, assumed aligned
    conf_a = mol_a.GetConformer(0).GetPositions()
    conf_b = mol_b.GetConformer(0).GetPositions()

    # note that >1 match possible here -- must pick minimum-cost match
    matches_a = mol_a.GetSubstructMatches(query)
    matches_b = mol_b.GetSubstructMatches(query)
    n_a, n_b = len(matches_a), len(matches_b)

    # cost[i, j] = sum_i distance(conf)
    cost = np.zeros((len(matches_a), len(matches_b)))
    for i, a in enumerate(matches_a):
        for j, b in enumerate(matches_b):
            cost[i, j] = np.linalg.norm(conf_a[np.array(a)] - conf_b[np.array(b)], axis=1).sum()

    # find (i,j) = argmin cost
    min_i, min_j = np.unravel_index(np.argmin(cost, axis=None), cost.shape)
    print(f'argmin of {n_a} x {n_b} cost matrix: {(min_i, min_j)} ')
    # TODO: maybe also print the difference between min(cost) and cost[0,0],
    #   to see how big of a difference it made to pick the default

    # TODO: is there a way to use the matching from MCS directly?

    # concatenate into (n_atoms, 2) array
    inds_a, inds_b = matches_a[min_i], matches_b[min_j]
    core = np.array([inds_a, inds_b]).T

    if not _check_core_map_distances(mol_a, mol_b, core, threshold):
        raise (AtomMappingError(f"not all mapped atoms are within {threshold:.3f}Å of each other!"))

    return core


from fe.utils import core_from_distances, simple_geometry_mapping


def _assert_core_reasonableness(mol_a, mol_b, core):
    # TODO move any useful run-time assertions from this script into tests/

    # bounds
    assert (max(core[:, 0]) < mol_a.GetNumAtoms())
    assert (max(core[:, 1]) < mol_b.GetNumAtoms())

    # uniqueness
    assert (len(set(core[:, 0])) == len(core))
    assert (len(set(core[:, 1])) == len(core))


def get_core_by_matching(mol_a, mol_b, threshold=1.0):
    """Only allow to map a pair of atoms together if their conformer coordinates are within threshold.

    Of the allowable core mappings, return the maximum-weight matching of maximal-cardinality,
        where weight(i,j) = threshold - distance(mol_a[i], mol_b[j])
    """
    core = core_from_distances(mol_a, mol_b, threshold)
    _assert_core_reasonableness(mol_a, mol_b, core)
    return core


def get_core_by_geometry(mol_a, mol_b, threshold=0.5):
    """Only allow to map a pair of atoms together if their conformer coordinates are within threshold.

    Of the allowable core mappings, return the one that contains only atom pairs (i, j)
    where i in mol_a has exactly one neighbor j in mol_b within threshold
    """
    core = simple_geometry_mapping(mol_a, mol_b, threshold)
    _assert_core_reasonableness(mol_a, mol_b, core)
    return core


def _get_unique_match(mol, core):
    matches = mol.GetSubstructMatches(core)
    assert len(matches) == 1
    return matches[0]


def get_core_by_smarts(mol_a, mol_b, core_smarts):
    """no atom mapping errors with this one, but the core size is smaller"""
    query = Chem.MolFromSmarts(core_smarts)
    return np.array([_get_unique_match(mol_a, query), _get_unique_match(mol_b, query)]).T


# 3.1. Define a custom maximum common substructure search method

def _identify_hub(transformation_sizes):
    return int(np.argmin(transformation_sizes.sum(0)))


def get_star_map(mols, path_to_results):
    mcs_s = compute_all_pairs_mcs(mols)
    transformation_sizes = compute_transformation_size_matrix(mols, mcs_s)

    hub_index = _identify_hub(transformation_sizes)
    hub = mols[hub_index]

    plot_transformation_sizes(transformation_sizes)
    plt.savefig(path_to_results.joinpath('transformation_sizes.png'), bbox_inches='tight')
    plt.close()

    # for each "spoke" in the star map, construct serializable transformation "hub -> spoke"
    others = list(mols)
    others.pop(hub_index)

    return hub, others


import matplotlib.pyplot as plt


def plot_transformation_sizes(transformation_sizes):
    plt.imshow(transformation_sizes)
    plt.xlabel('molecule index')
    plt.ylabel('molecule index')
    plt.title('"size of transformation"\n$(n_A - n_{MCS}) + (n_B - n_{MCS})$')

    plt.tight_layout()

    plt.colorbar()
