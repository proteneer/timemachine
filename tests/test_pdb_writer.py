import pytest
import numpy as np

from tempfile import NamedTemporaryFile

from fe.pdb_writer import PDBWriter, convert_single_topology_mols
from fe.topology import SingleTopology, DualTopology
from timemachine.md import builders

from testsystems.relative import hif2a_ligand_pair


def test_write_single_topology_frame():
    top = hif2a_ligand_pair.top
    assert isinstance(top, SingleTopology)

    ff_params = hif2a_ligand_pair.top.ff.get_ordered_params()

    solvent_system, solvent_coords, solvent_box, solvent_top = builders.build_water_system(4.0)

    unbound_potentials, sys_params, masses, coords = hif2a_ligand_pair.prepare_host_edge(
        ff_params, solvent_system, solvent_coords
    )

    coords *= 10  # nm to angstroms
    with NamedTemporaryFile(suffix=".pdb") as temp:
        writer = PDBWriter([solvent_top, top.mol_a, top.mol_b], temp.name)
        with pytest.raises(ValueError):
            # Should fail, as incorrect number of coords
            writer.write_frame(coords)
        ligand_coords = convert_single_topology_mols(coords[len(solvent_coords) :], top)
        writer.write_frame(np.concatenate((coords[: len(solvent_coords)], ligand_coords), axis=0))
        writer.close()
