"""Tests related to timemachine when there are no GPUs"""

import numpy as np
import pytest

from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import custom_ops
from timemachine.md import builders

# Run tests in the no-gpu, nightly tests
pytestmark = [pytest.mark.nogpu, pytest.mark.nightly]


def test_no_gpu_raises_exception():
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    solvent_system, solvent_coords, solvent_box, top = builders.build_water_system(3.0, ff.water_ff)

    host_fns, _ = openmm_deserializer.deserialize_system(solvent_system, cutoff=1.2)

    with pytest.raises(custom_ops.InvalidHardware, match="Invalid Hardware - Code "):
        host_fns[0].to_gpu(np.float32)
