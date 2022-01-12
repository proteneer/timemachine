from unittest import TestCase

from fe.estimator_abfe import SimulationResult
from fe.frames import endpoint_frames_only, all_frames


def make_dummy_sim_result(val):
    return SimulationResult(
        xs=[val],
        boxes=[val],
        lambda_us=[val],
    )


class FrameFiltersTestCase(TestCase):
    def test_endpoint_only_filter(self):

        res = list(endpoint_frames_only([]))
        assert len(res) == 0

        mock_sims = [make_dummy_sim_result("a")]
        res = list(endpoint_frames_only(mock_sims))
        assert len(res) == 1
        assert res[0] == (0, mock_sims[0])

        mock_sims = [make_dummy_sim_result(x) for x in ("a", "b", "c")]
        res = list(endpoint_frames_only(mock_sims))
        assert len(res) == 2
        assert res[0] == (0, mock_sims[0])
        assert res[1] == (2, mock_sims[2])

    def test_all_frames(self):

        res = list(all_frames([]))
        assert len(res) == 0

        mock_sims = [make_dummy_sim_result("a")]
        res = list(all_frames(mock_sims))
        assert len(res) == 1
        assert res[0] == (0, mock_sims[0])

        mock_sims = [make_dummy_sim_result(x) for x in ("a", "b", "c")]
        res = list(all_frames(mock_sims))
        assert len(res) == 3
        assert res[0] == (0, mock_sims[0])
        assert res[1] == (1, mock_sims[1])
        assert res[2] == (2, mock_sims[2])
