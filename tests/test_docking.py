"""
Tests for the timemachine/docking/ files
"""

import unittest
from pathlib import Path
from testsystems.relative import hif2a_ligand_pair
from docking import pose_dock, dock_and_equilibrate, relative_docking


class TestDocking(unittest.TestCase):
    def test_pose_dock(self):
        """Tests basic functionality of pose_dock"""
        guests_sdfile = str(
            Path(__file__)
            .resolve()
            .parent.joinpath("data", "ligands_40__first-two-ligs.sdf")
        )
        host_pdbfile = str(
            Path(__file__).resolve().parent.joinpath("data", "hif2a_nowater_min.pdb")
        )
        transition_type = "insertion"
        n_steps = 1001
        transition_steps = 500
        max_lambda = 0.5
        outdir = "test_pd_outdir"

        pose_dock.pose_dock(
            guests_sdfile,
            host_pdbfile,
            transition_type,
            n_steps,
            transition_steps,
            max_lambda,
            outdir,
        )

        self.assertTrue(Path("test_pd_outdir/338/").exists())
        self.assertTrue(Path("test_pd_outdir/338/338_pd_1000_host.pdb").exists())
        self.assertTrue(Path("test_pd_outdir/338/338_pd_1000_guest.sdf").exists())
        self.assertTrue(Path("test_pd_outdir/43/").exists())
        self.assertTrue(Path("test_pd_outdir/43/43_pd_1000_host.pdb").exists())
        self.assertTrue(Path("test_pd_outdir/43/43_pd_1000_guest.sdf").exists())

        for f in Path("test_pd_outdir").glob("*/*.*"):
            Path.unlink(f)
        for d in Path("test_pd_outdir").glob("*"):
            d.rmdir()
        Path("test_pd_outdir").rmdir()


    def test_dock_and_equilibrate(self):
        """Tests basic functionality of dock_and_equilibrate"""
        host_pdbfile = str(
            Path(__file__).resolve().parent.joinpath("data", "hif2a_nowater_min.pdb")
        )
        guests_sdfile = str(
            Path(__file__)
            .resolve()
            .parent.joinpath("data", "ligands_40__first-two-ligs.sdf")
        )
        max_lambda = 0.25
        insertion_steps = 501
        eq_steps = 1501
        outdir = 'test_de_outdir'
        dock_and_equilibrate.dock_and_equilibrate(
            host_pdbfile,
            guests_sdfile,
            max_lambda,
            insertion_steps,
            eq_steps,
            outdir,
        )
        self.assertTrue(Path("test_de_outdir/338/").exists())
        self.assertTrue(Path(f"test_de_outdir/338/338_ins_{insertion_steps-1}_host.pdb").exists())
        self.assertTrue(Path("test_de_outdir/338/338_eq_1000_guest.sdf").exists())
        self.assertTrue(Path("test_de_outdir/43/").exists())
        self.assertTrue(Path(f"test_de_outdir/43/43_ins_{insertion_steps-1}_host.pdb").exists())
        self.assertTrue(Path("test_de_outdir/43/43_eq_1000_guest.sdf").exists())

        for f in Path("test_de_outdir").glob("*/*.*"):
            Path.unlink(f)
        for d in Path("test_de_outdir").glob("*"):
            d.rmdir()
        Path("test_de_outdir").rmdir()


    def test_rigorous_work(self):
        """Tests basic functionality of rigorous_work"""
        raise NotImplementedError()


    def test_relative_docking(self):
        """Tests basic functionality of relative_docking"""
        # fetch mol_a, mol_b, core, forcefield from testsystem
        mol_a, mol_b, core = (
            hif2a_ligand_pair.mol_a,
            hif2a_ligand_pair.mol_b,
            hif2a_ligand_pair.core,
        )
        host_pdbfile = str(
            Path(__file__).resolve().parent.joinpath("data", "hif2a_nowater_min.pdb")
        )
        works = relative_docking.do_relative_docking(
            host_pdbfile,
            mol_a,
            mol_b,
            core,
        )
        self.assertTrue("protein" in works)
        self.assertTrue("solvent" in works)
        self.assertTrue(len(works["protein"]) == 10)
        self.assertTrue(len(works["solvent"]) == 10)


if __name__ == "__main__":
    unittest.main()
