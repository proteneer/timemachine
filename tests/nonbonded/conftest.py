import numpy as np
import pytest

from timemachine.fe.utils import to_md_units
from timemachine.lib import potentials
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

    nonbonded_fn = None
    for f in host_fns:
        if isinstance(f, potentials.Nonbonded):
            nonbonded_fn = f

    assert nonbonded_fn is not None
    return nonbonded_fn


@pytest.fixture()
def example_conf(_example_system):
    _, _, host_conf, _ = _example_system
    return np.array([[to_md_units(x), to_md_units(y), to_md_units(z)] for x, y, z in host_conf])


@pytest.fixture()
def example_box(_example_system):
    _, _, _, box = _example_system
    return box


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(2022)
    yield


@pytest.fixture()
def rng():
    return np.random.default_rng(2022)
