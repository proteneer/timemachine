from tempfile import NamedTemporaryFile

import numpy as np
import pytest
from openmm.app import PDBxFile

from timemachine.fe.cif_writer import CIFWriter, convert_single_topology_mols
from timemachine.fe.single_topology import SingleTopology
from timemachine.fe.utils import get_romol_conf
from timemachine.ff import Forcefield
from timemachine.md import builders
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology
from timemachine.utils import path_to_internal_file

pytestmark = [pytest.mark.nocuda]


def test_write_single_topology_frame():
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_default()
    top = SingleTopology(mol_a, mol_b, core, forcefield)
    host_config = builders.build_water_system(4.0, forcefield.water_ff, mols=[mol_a, mol_b])

    with NamedTemporaryFile(suffix=".cif") as temp:
        writer = CIFWriter([host_config.omm_topology, mol_a, mol_b], temp.name)

        ligand_coords = top.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))

        bad_coords = np.concatenate([host_config.conf, ligand_coords])

        with pytest.raises(ValueError):
            # Should fail, as incorrect number of coords
            bad_coords = bad_coords * 10
            writer.write_frame(bad_coords)

        good_coords = np.concatenate([host_config.conf, convert_single_topology_mols(ligand_coords, top)], axis=0)

        # tbd replace with atom map mixin
        writer.write_frame(good_coords * 10)
        writer.close()
        cif = PDBxFile(temp.name)
        assert cif.getNumFrames() == 1
        assert cif.getPositions(asNumpy=True).shape == good_coords.shape


@pytest.mark.parametrize("n_frames", [1, 4, 5])
def test_cif_writer_context(n_frames):
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()

    # test vacuum
    with NamedTemporaryFile(suffix=".cif") as temp:
        good_coords = np.concatenate([get_romol_conf(mol_a), get_romol_conf(mol_b)], axis=0)
        with CIFWriter([mol_a, mol_b], temp.name) as writer:
            for _ in range(n_frames):
                writer.write_frame(good_coords * 10)
        cif = PDBxFile(temp.name)
        assert cif.getNumFrames() == n_frames
        assert cif.getPositions(asNumpy=True).shape == good_coords.shape


@pytest.mark.parametrize("n_frames", [1, 4, 5])
def test_cif_writer(n_frames):
    ff = Forcefield.load_default()

    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()

    # test vacuum
    with NamedTemporaryFile(suffix=".cif") as temp:
        writer = CIFWriter([mol_a, mol_b], temp.name)
        good_coords = np.concatenate([get_romol_conf(mol_a), get_romol_conf(mol_b)], axis=0)
        for _ in range(n_frames):
            writer.write_frame(good_coords * 10)
        writer.close()
        cif = PDBxFile(temp.name)
        assert cif.getNumFrames() == n_frames
        assert cif.getPositions(asNumpy=True).shape == good_coords.shape
        np.testing.assert_allclose(cif.getPositions(asNumpy=True), good_coords)

    solvent_host_config = builders.build_water_system(4.0, ff.water_ff, mols=[mol_a, mol_b])

    # test solvent
    with NamedTemporaryFile(suffix=".cif") as temp:
        writer = CIFWriter([solvent_host_config.omm_topology, mol_a, mol_b], temp.name)
        good_coords = np.concatenate([solvent_host_config.conf, get_romol_conf(mol_a), get_romol_conf(mol_b)], axis=0)
        for _ in range(n_frames):
            writer.write_frame(good_coords * 10)
        writer.close()
        cif = PDBxFile(temp.name)
        assert cif.getNumFrames() == n_frames
        assert cif.getPositions(asNumpy=True).shape == good_coords.shape
        # Tolerance is difference due to shifting water coords such that the mols are at the center
        np.testing.assert_allclose(cif.getPositions(asNumpy=True), good_coords, atol=1e-5)

    # test complex
    with path_to_internal_file("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        complex_host_config = builders.build_protein_system(str(path_to_pdb), ff.protein_ff, ff.water_ff)

        with NamedTemporaryFile(suffix=".cif") as temp:
            writer = CIFWriter([complex_host_config.omm_topology, mol_a, mol_b], temp.name)
            good_coords = np.concatenate(
                [complex_host_config.conf, get_romol_conf(mol_a), get_romol_conf(mol_b)], axis=0
            )
            for _ in range(n_frames):
                writer.write_frame(good_coords * 10)
            writer.close()
            cif = PDBxFile(temp.name)
            assert cif.getNumFrames() == n_frames
            assert cif.getPositions(asNumpy=True).shape == good_coords.shape
            np.testing.assert_allclose(cif.getPositions(asNumpy=True), good_coords)
