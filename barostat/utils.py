import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist


def compute_box_volume(box: np.ndarray) -> float:
    return np.linalg.det(box)


def compute_box_center(box: np.ndarray) -> np.ndarray:
    # assume axis-aligned box (nothing off diagonal)
    assert np.linalg.norm(box - np.diag(np.diag(box))) == 0

    return np.sum(box / 2, axis=0)


def get_group_indices(harmonic_bond_potential):
    # read off topology from harmonic bond indices
    # NOTE: this assumes all bonds are represented by harmonic bond force
    bond_list = list(map(tuple, harmonic_bond_potential.get_idxs()))
    # TODO: if we add HBond constraints, be sure to add these to the bond_list!
    alchemical_topology = nx.Graph(bond_list)
    connected_components = list(map(list, nx.algorithms.connected_components(alchemical_topology)))

    return connected_components


def merge_big_groups(group_indices):
    """
    assume any molecules with > 3 atoms are the protein and the ligand,
    and treat the protein:ligand complex as a unit
    """

    molecule_sizes = np.array(list(map(len, group_indices)))

    protein_and_ligand_mol_inds = np.where(molecule_sizes > 3)[0]
    print(
        f'merging {len(protein_and_ligand_mol_inds)} connected components, of sizes {molecule_sizes[protein_and_ligand_mol_inds]}')

    # waters, and possibly ions if present
    other_mol_inds = np.where(molecule_sizes <= 3)[0]

    protein_ligand_group = np.hstack([group_indices[i] for i in protein_and_ligand_mol_inds])
    merged_group_indices = [protein_ligand_group] + [group_indices[i] for i in other_mol_inds]

    return merged_group_indices


def compute_intramolecular_distances(coords, group_indices):
    return [pdist(coords[inds]) for inds in group_indices]
