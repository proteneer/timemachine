"""
Tests for the timemachine/docking/ files
"""
import tempfile
import unittest
from importlib import resources
from pathlib import Path

from rdkit import Chem

from docking import dock_and_equilibrate, pose_dock, relative_docking, rigorous_work
from timemachine.testsystems.relative import hif2a_ligand_pair


def get_hif2a_ligands_as_sdf_file(num_mols: int) -> tempfile.NamedTemporaryFile:
    mols = []
    with resources.path("timemachine.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        suppl = Chem.SDMolSupplier(str(path_to_ligand), removeHs=False)
        for _ in range(num_mols):
            mols.append(next(suppl))
    temp_sdf = tempfile.NamedTemporaryFile(suffix=".sdf")
    with Chem.SDWriter(temp_sdf.name) as writer:
        for mol in mols:
            writer.write(mol)
    return temp_sdf


class TestDocking(unittest.TestCase):
    def test_pose_dock(self):
        """Tests basic functionality of pose_dock"""
        temp_sdf = get_hif2a_ligands_as_sdf_file(1)
        host_pdbfile = str(Path(__file__).resolve().parent.parent.joinpath("tests", "data", "hif2a_nowater_min.pdb"))
        transition_type = "insertion"
        n_steps = 1001
        transition_steps = 500
        max_lambda = 0.5
        with tempfile.TemporaryDirectory() as outdir:

            pose_dock.pose_dock(
                host_pdbfile,
                temp_sdf.name,
                transition_type,
                n_steps,
                transition_steps,
                max_lambda,
                outdir,
            )

            expected_output = [
                f"{outdir}/338/",
                f"{outdir}/338/338_pd_1000_host.pdb",
                f"{outdir}/338/338_pd_1000_guest.sdf",
            ]

            for f in expected_output:
                self.assertTrue(Path(f).exists())

    def test_dock_and_equilibrate(self):
        """Tests basic functionality of dock_and_equilibrate"""
        host_pdbfile = str(Path(__file__).resolve().parent.parent.joinpath("tests", "data", "hif2a_nowater_min.pdb"))
        temp_sdf = get_hif2a_ligands_as_sdf_file(1)
        max_lambda = 0.25
        insertion_steps = 501
        eq_steps = 1501

        with tempfile.TemporaryDirectory() as outdir:
            dock_and_equilibrate.dock_and_equilibrate(
                host_pdbfile, temp_sdf.name, max_lambda, insertion_steps, eq_steps, outdir
            )

            expected_output = [
                f"{outdir}/338/",
                f"{outdir}/338/338_ins_{insertion_steps-1}_host.pdb",
                f"{outdir}/338/338_eq_1000_guest.sdf",
            ]

            for f in expected_output:
                self.assertTrue(Path(f).exists())

    def test_rigorous_work(self):
        """Tests basic functionality of rigorous_work"""
        host_pdbfile = str(Path(__file__).resolve().parent.parent.joinpath("tests", "data", "hif2a_nowater_min.pdb"))
        temp_sdf = get_hif2a_ligands_as_sdf_file(1)
        num_deletions = 10
        deletion_steps = 501
        with tempfile.TemporaryDirectory() as outdir:
            all_works = rigorous_work.calculate_rigorous_work(
                host_pdbfile, temp_sdf.name, outdir, num_deletions, deletion_steps
            )

            self.assertTrue("338" in all_works)
            self.assertTrue("protein" in all_works["338"])
            self.assertTrue("solvent" in all_works["338"])
            self.assertEqual(len(all_works["338"]["protein"]), 10)

            expected_output = [
                f"{outdir}/338/",
                f"{outdir}/338/338_solvent-ins_500_host.pdb",
                f"{outdir}/338/338_protein-eq1_5000_guest.sdf",
                f"{outdir}/338/338_protein-eq2_05000_guest.sdf",
            ]

            for f in expected_output:
                self.assertTrue(Path(f).exists())

    def test_relative_docking(self):
        """Tests basic functionality of relative_docking"""
        # fetch mol_a, mol_b, core, forcefield from testsystem
        mol_a, mol_b, core = (
            hif2a_ligand_pair.mol_a,
            hif2a_ligand_pair.mol_b,
            hif2a_ligand_pair.top.core,
        )
        host_pdbfile = str(Path(__file__).resolve().parent.parent.joinpath("tests", "data", "hif2a_nowater_min.pdb"))
        num_switches = 10
        transition_steps = 501
        works = relative_docking.do_relative_docking(host_pdbfile, mol_a, mol_b, core, num_switches, transition_steps)
        self.assertTrue("protein" in works)
        self.assertTrue("solvent" in works)
        self.assertEqual(len(works["protein"]), num_switches)
        self.assertEqual(len(works["solvent"]), num_switches)


if __name__ == "__main__":
    unittest.main()
