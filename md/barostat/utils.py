import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist
from typing import List
from timemachine.lib.potentials import HarmonicBond

from md.barostat.moves import CoordsAndBox, MonteCarloBarostat
from md.ensembles import NPTEnsemble
from md.thermostat.utils import run_thermostatted_md
from tqdm import tqdm
from time import time
from typing import Tuple, Dict


def compute_box_volume(box: np.ndarray) -> float:
    assert box.shape == (3, 3)
    return np.linalg.det(box)


def compute_box_center(box: np.ndarray) -> np.ndarray:
    # assume axis-aligned box (nothing off diagonal)
    assert box.shape == (3, 3)
    assert np.linalg.norm(box - np.diag(np.diag(box))) == 0

    center = np.sum(box / 2, axis=0)

    assert center.shape == (3,)

    return center


def get_group_indices(harmonic_bond_potential: HarmonicBond) -> List[np.array]:
    # read off topology from harmonic bond indices
    # NOTE: this assumes all bonds are represented by harmonic bond force
    bond_list = list(map(tuple, harmonic_bond_potential.get_idxs()))
    # TODO: if we add HBond constraints, be sure to add these to the bond_list!
    alchemical_topology = nx.Graph(bond_list)
    connected_components = list(map(list, nx.algorithms.connected_components(alchemical_topology)))

    return connected_components


def merge_big_groups(group_indices: List[np.array]) -> List[np.array]:
    """
    assume any molecules with > 3 atoms are the protein and the ligand,
    and treat the protein:ligand complex as a unit
    """

    molecule_sizes = np.array(list(map(len, group_indices)))

    protein_and_ligand_mol_inds = np.where(molecule_sizes > 3)[0]
    n_components, component_sizes = len(protein_and_ligand_mol_inds), molecule_sizes[protein_and_ligand_mol_inds]
    print(f'merging {n_components} connected components, of sizes {component_sizes}')

    # waters, and possibly ions if present
    other_mol_inds = np.where(molecule_sizes <= 3)[0]

    protein_ligand_group = np.hstack([group_indices[i] for i in protein_and_ligand_mol_inds])
    merged_group_indices = [protein_ligand_group] + [group_indices[i] for i in other_mol_inds]

    return merged_group_indices


def compute_intramolecular_distances(coords: np.array, group_indices: List[np.array]) -> List[np.array]:
    """pairwise distances within each group"""
    return [pdist(coords[inds]) for inds in group_indices]


def simulate_npt_traj(ensemble: NPTEnsemble, integrator_impl, barostat: MonteCarloBarostat,
                      coords: np.array, box: np.array, velocities: np.array,
                      lam=1.0, n_moves=1000, barostat_interval=5) -> Tuple[np.array, np.array, Dict]:
    """TODO: replace with more modular design: composition of [MDMove, MCBarostatMove]"""
    barostat.reset()

    # alternate between thermostat moves and barostat moves
    traj = [CoordsAndBox(coords, box)]
    volume_traj = [compute_box_volume(traj[0].box)]
    proposal_scale_traj = [barostat.max_delta_volume]

    trange = tqdm(range(n_moves))

    bound_impls = ensemble.potential_energy.all_impls

    v_t = velocities.copy()
    for _ in trange:
        t0 = time()

        # MDMove
        x_0, v_0, box = traj[-1].coords, v_t.copy(), traj[-1].box
        x_t, v_t = run_thermostatted_md(
            integrator_impl, bound_impls, x_0, box, v_0, lam, n_steps=barostat_interval)
        after_nvt = CoordsAndBox(x_t, box)

        t1 = time()

        # MCBarostatMove
        after_npt = barostat.move(after_nvt)

        t2 = time()

        # accumulate result trajectories
        traj.append(after_npt)
        volume_traj.append(compute_box_volume(after_npt.box))
        proposal_scale_traj.append(barostat.max_delta_volume)

        # informative progress bar
        trange.set_postfix(volume=f'{volume_traj[-1]:.3f}',
                           acceptance_fraction=f'{barostat.acceptance_fraction:.3f}',
                           md_proposal_time=f'{(t1 - t0):.3f}s',
                           barostat_proposal_time=f'{(t2 - t1):.3f}s',
                           proposal_scale=f'{barostat.max_delta_volume:.3f}',
                           )

    # TODO: make this an MDTraj trajectory?
    x_traj = np.array([snapshot.coords for snapshot in traj])
    box_traj = np.array([snapshot.box for snapshot in traj])

    volume_traj = np.array(volume_traj)
    proposal_scale_traj = np.array(proposal_scale_traj)

    extras = dict(volume_traj=volume_traj, proposal_scale_traj=proposal_scale_traj)

    return x_traj, box_traj, extras
