# construct a relative transformation

from importlib import resources

import numpy as np
from rdkit import Chem

from timemachine.fe import free_energy, topology
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import Forcefield

DEFAULT_FORCEFIELD = "smirnoff_1_1_0_ccc.py"


def prepare_vacuum_edge(top):
    """
    Prepare a vacuum edge for a relative transformation.

    Parameters
    ----------
    top: Union[topology.SingleTopology, topology.DualTopology]
        Topology to use

    Returns
    -------
        3-tuple
            unbound_potentials, system_params, combined_masses, combined_coords
    """
    ligand_masses_a = [a.GetMass() for a in top.mol_a.GetAtoms()]
    ligand_masses_b = [b.GetMass() for b in top.mol_b.GetAtoms()]

    ligand_coords_a = get_romol_conf(top.mol_a)
    ligand_coords_b = get_romol_conf(top.mol_b)

    final_params, final_potentials = free_energy.BaseFreeEnergy._get_system_params_and_potentials(
        top.ff.get_ordered_params(), top
    )

    combined_masses = np.mean(top.interpolate_params(ligand_masses_a, ligand_masses_b), axis=0)
    combined_coords = np.mean(top.interpolate_params(ligand_coords_a, ligand_coords_b), axis=0)

    return final_potentials, final_params, combined_masses, combined_coords


def _setup_hif2a_ligand_topology(ff=DEFAULT_FORCEFIELD):
    """Manually constructed atom map

    TODO: replace this with a testsystem class similar to those used in openmmtools
    """

    forcefield = Forcefield.load_from_file(ff)

    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        suppl = Chem.SDMolSupplier(str(path_to_ligand), removeHs=False)

    all_mols = [x for x in suppl]
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

    return topology.SingleTopology(mol_a, mol_b, core, forcefield)


def setup_hif2a_ligand_pair(ff=DEFAULT_FORCEFIELD):
    single_topology = _setup_hif2a_ligand_topology(ff)
    return free_energy.RelativeFreeEnergy(single_topology)


def _setup_hif2a_vacuum(ff=DEFAULT_FORCEFIELD):
    """
    Prepares the vacuum system.
    Parameters
    ----------
    ff: str
        Name of forcefield file.

    Returns
    -------
    4 tuple
        unbound_potentials, system_parameters, combined_masses, combined_coords
    """
    top = _setup_hif2a_ligand_topology()
    return prepare_vacuum_edge(top)


hif2a_ligand_pair_vacuum_edge = _setup_hif2a_vacuum()
hif2a_ligand_pair = setup_hif2a_ligand_pair()
