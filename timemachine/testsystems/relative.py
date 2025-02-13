# construct a relative transformation


import numpy as np

from timemachine.constants import DEFAULT_ATOM_MAPPING_KWARGS
from timemachine.fe import atom_mapping
from timemachine.fe.rbfe import setup_initial_states
from timemachine.fe.single_topology import SingleTopology
from timemachine.fe.utils import get_romol_conf, read_sdf
from timemachine.ff import Forcefield
from timemachine.utils import path_to_internal_file


def get_hif2a_ligand_pair_single_topology():
    """Return two ligands from hif2a and the manually specified atom mapping"""

    with path_to_internal_file("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        all_mols = read_sdf(str(path_to_ligand))

    mol_a = all_mols[1]
    mol_b = all_mols[4]

    core = np.array(
        [
            [0, 0],
            [2, 2],
            [1, 1],
            [6, 6],
            [5, 5],
            [4, 4],
            [3, 3],
            [15, 16],
            [16, 17],
            [17, 18],
            [18, 19],
            [19, 20],
            [20, 21],
            [32, 30],
            [26, 25],
            [27, 26],
            [7, 7],
            [8, 8],
            [9, 9],
            [10, 10],
            [29, 11],
            [11, 12],
            [12, 13],
            [14, 15],
            [31, 29],
            [13, 14],
            [23, 24],
            [30, 28],
            [28, 27],
            [21, 22],
        ]
    )
    return mol_a, mol_b, core


def get_hif2a_ligand_pair_single_topology_chiral_volume():
    """hif2_pair with a chiral CF3 (mol_a) morphed to achiral NH2 (mol_b)"""

    with path_to_internal_file("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        all_mols = read_sdf(str(path_to_ligand))

    mol_a = all_mols[11]
    mol_b = all_mols[-7]

    core = atom_mapping.get_cores(
        mol_a,
        mol_b,
        **DEFAULT_ATOM_MAPPING_KWARGS,
    )[0]

    return mol_a, mol_b, core


def get_hif2a_ligand_pair(src_idx, dst_idx):
    """hif2_pair with a chiral CF3 (mol_a) morphed to achiral NH2 (mol_b)"""

    with path_to_internal_file("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        all_mols = read_sdf(str(path_to_ligand))

    mol_a = all_mols[src_idx]
    mol_b = all_mols[dst_idx]

    core = atom_mapping.get_cores(
        mol_a,
        mol_b,
        **DEFAULT_ATOM_MAPPING_KWARGS,
    )[0]

    return mol_a, mol_b, core


def get_relative_hif2a_in_vacuum():
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_default()
    rfe = SingleTopology(mol_a, mol_b, core, ff)

    temperature = 300
    seed = 2022
    lam = 0.5
    host = None  # vacuum
    initial_states = setup_initial_states(rfe, host, temperature, [lam], seed)
    potentials = initial_states[0].potentials
    sys_params = [np.array(u.params, dtype=np.float64) for u in potentials]
    coords = rfe.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))
    masses = np.array(rfe.combine_masses())
    return potentials, sys_params, coords, masses
