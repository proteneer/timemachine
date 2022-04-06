from tempfile import NamedTemporaryFile

import numpy as np
import pytest

from timemachine.fe.pdb_writer import PDBWriter, convert_single_topology_mols
from timemachine.fe.topology import SingleTopology
from timemachine.md import builders
from timemachine.testsystems.relative import hif2a_ligand_pair


def test_write_single_topology_frame():
    top = hif2a_ligand_pair.top
    assert isinstance(top, SingleTopology)

    ff_params = hif2a_ligand_pair.top.ff.get_ordered_params()

    solvent_system, solvent_coords, solvent_box, solvent_top = builders.build_water_system(4.0)

    unbound_potentials, sys_params, masses = hif2a_ligand_pair.prepare_host_edge(ff_params, solvent_system)
    coords = hif2a_ligand_pair.prepare_combined_coords(solvent_coords)

    coords *= 10  # nm to angstroms
    with NamedTemporaryFile(suffix=".pdb") as temp:
        writer = PDBWriter([solvent_top, top.mol_a, top.mol_b], temp.name)
        with pytest.raises(ValueError):
            # Should fail, as incorrect number of coords
            writer.write_frame(coords)
        ligand_coords = convert_single_topology_mols(coords[len(solvent_coords) :], top)
        writer.write_frame(np.concatenate((coords[: len(solvent_coords)], ligand_coords), axis=0))
        writer.close()
