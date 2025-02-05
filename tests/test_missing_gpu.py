"""Tests related to timemachine when there are no GPUs"""

import numpy as np
import pytest

from timemachine.ff import Forcefield
from timemachine.lib import custom_ops
from timemachine.md import builders

# Run tests in the no-gpu, nightly tests
pytestmark = [pytest.mark.nogpu, pytest.mark.nightly]


def test_no_gpu_raises_exception():
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    host_config = builders.build_water_system(3.0, ff.water_ff)

    host_fns = host_config.host_system.get_U_fns()

    with pytest.raises(custom_ops.InvalidHardware, match="Invalid Hardware - Code "):
        host_fns[0].to_gpu(np.float32)
