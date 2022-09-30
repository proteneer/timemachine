from tempfile import NamedTemporaryFile

import numpy as np
import pytest

from timemachine.constants import DEFAULT_FF
from timemachine.fe.pdb_writer import PDBWriter, convert_single_topology_mols
from timemachine.fe.single_topology_v3 import SingleTopologyV3
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import Forcefield
from timemachine.md import builders
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


@pytest.mark.nogpu
def test_write_single_topology_frame():
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file(DEFAULT_FF)
    top = SingleTopologyV3(mol_a, mol_b, core, forcefield)

    solvent_system, solvent_coords, solvent_box, solvent_top = builders.build_water_system(4.0)
    coords = top.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))
    coords = np.concatenate([solvent_coords, coords], axis=0)
    coords *= 10  # nm to angstroms
    with NamedTemporaryFile(suffix=".pdb") as temp:
        writer = PDBWriter([solvent_top, top.mol_a, top.mol_b], temp.name)
        with pytest.raises(ValueError):
            # Should fail, as incorrect number of coords
            writer.write_frame(coords)
        ligand_coords = convert_single_topology_mols(coords[len(solvent_coords) :], top)
        writer.write_frame(np.concatenate((coords[: len(solvent_coords)], ligand_coords), axis=0))
        writer.close()
