from dataclasses import dataclass

import numpy as np
import openmm.app.topology as ommt
import openmm.openmm as omm
from numpy.typing import NDArray


@dataclass(frozen=True)
class HostConfig:
    omm_system: omm.System
    conf: NDArray[np.float64]
    box: NDArray[np.float64]
    num_water_atoms: int
    omm_topology: ommt.Topology
