from tempfile import NamedTemporaryFile

import numpy as np
import pytest

from timemachine.fe.pdb_writer import PDBWriter, convert_single_topology_mols
from timemachine.fe.utils import get_romol_conf
from timemachine.md import builders
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


@pytest.mark.nogpu
def test_write_single_topology_frame():
    st = get_hif2a_ligand_pair_single_topology()
    solvent_system, solvent_coords, solvent_box, solvent_top = builders.build_water_system(4.0)

    with NamedTemporaryFile(suffix=".pdb") as temp:
        writer = PDBWriter([solvent_top, st.mol_a, st.mol_b], temp.name)

        ligand_coords = st.combine_confs(get_romol_conf(st.mol_a), get_romol_conf(st.mol_b))

        bad_coords = np.concatenate([solvent_coords, ligand_coords])

        with pytest.raises(ValueError):
            # Should fail, as incorrect number of coords
            bad_coords = bad_coords * 10
            writer.write_frame(bad_coords)

        good_coords = np.concatenate([solvent_coords, convert_single_topology_mols(ligand_coords, st)], axis=0)

        # tbd replace with atom map mixin
        writer.write_frame(good_coords)
        writer.close()
