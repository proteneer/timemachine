from copy import deepcopy

import numpy as np
import pytest

from timemachine import potentials
from timemachine.testsystems.dhfr import setup_dhfr


# NOTE: This fixture is module-scoped so that it is only evaluated
# once and cached for efficiency. To prevent unintended mutation, this
# fixture shouldn't be used directly
@pytest.fixture(scope="module")
def _example_system():
    host_fns, host_masses, host_coords, box = setup_dhfr()
    return host_fns, host_masses, host_coords, box


@pytest.fixture()
def example_nonbonded_potential(_example_system):
    host_fns, _, _, _ = _example_system
    nonbonded_bp = next(bp for bp in host_fns if isinstance(bp.potential, potentials.Nonbonded))
    nonbonded_bp.cutoff = 1.2  # Need to set the cutoff to 1.2 to make the switch func happy
    return deepcopy(nonbonded_bp)


@pytest.fixture()
def example_conf(_example_system):
    _, _, host_conf, _ = _example_system
    return np.array(host_conf)


@pytest.fixture()
def example_box(_example_system):
    _, _, _, box = _example_system
    return np.array(box)


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(2022)
    yield


@pytest.fixture()
def rng():
    return np.random.default_rng(2022)
