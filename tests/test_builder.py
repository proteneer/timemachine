from unittest import TestCase

from md.builders import build_protein_system, build_water_system


class ProteinSystemTestCase(TestCase):

    def test_protein_system_prepare_prepared_system(self):
        _, unprepared_coords, _, _, _, _ = build_protein_system("tests/data/hif2a_nowater_min.pdb", prepare=False)

        _, prepared_coords, _, _, _, _ = build_protein_system("tests/data/hif2a_nowater_min.pdb", prepare=False)
        self.assertEqual(unprepared_coords.shape, prepared_coords.shape)

    def test_protein_system_preparation(self):
        _, unprepared_coords, _, _, _, _ = build_protein_system("datasets/fep-benchmark/hif2a/5tbm_prepared.pdb", prepare=False)

        _, prepared_coords, _, _, _, _ = build_protein_system("datasets/fep-benchmark/hif2a/5tbm_prepared.pdb", prepare=True)
        self.assertNotEqual(unprepared_coords.shape, prepared_coords.shape)
        self.assertEqual(unprepared_coords.shape[1], prepared_coords.shape[1])
        self.assertGreater(unprepared_coords.shape[0], prepared_coords.shape[0])