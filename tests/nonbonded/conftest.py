import numpy as np
import pytest
from simtk.openmm import app

from timemachine.fe.utils import to_md_units
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import potentials

# NOTE: For efficiency, we use module-scoped fixtures for expensive
# setup. To prevent unintended mutation, these shouldn't be used in
# tests directly. Instead, they are wrapped below by function-scoped
# fixtures that return copies of the data.


@pytest.fixture(scope="module")
def _example_system():
    pdb_path = "tests/data/5dfr_solv_equil.pdb"
    host_pdb = app.PDBFile(pdb_path)
    ff = app.ForceField("amber99sbildn.xml", "tip3p.xml")
    return (
        ff.createSystem(host_pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False),
        host_pdb.positions,
        host_pdb.topology.getPeriodicBoxVectors(),
    )


@pytest.fixture(scope="module")
def _example_nonbonded_potential(_example_system):
    host_system, _, _ = _example_system
    host_fns, _ = openmm_deserializer.deserialize_system(host_system, cutoff=1.0)

    nonbonded_fn = None
    for f in host_fns:
        if isinstance(f, potentials.Nonbonded):
            nonbonded_fn = f

    assert nonbonded_fn is not None
    return nonbonded_fn


@pytest.fixture()
def _example_nonbonded_params(_example_nonbonded_potential):
    return _example_nonbonded_potential.params


@pytest.fixture(scope="module")
def _example_conf(_example_system):
    _, host_conf, _ = _example_system
    return np.array([[to_md_units(x), to_md_units(y), to_md_units(z)] for x, y, z in host_conf])


@pytest.fixture(scope="function", autouse=True)
def set_random_seed():
    np.random.seed(2022)
    yield


@pytest.fixture(scope="function")
def rng():
    return np.random.default_rng(2022)


@pytest.fixture(scope="function")
def example_nonbonded_params(_example_nonbonded_params):
    return np.array(_example_nonbonded_params)


@pytest.fixture()
def example_nonbonded_exclusion_idxs(_example_nonbonded_potential):
    return _example_nonbonded_potential.get_exclusion_idxs()


@pytest.fixture()
def example_nonbonded_exclusion_scales(_example_nonbonded_potential):
    return _example_nonbonded_potential.get_scale_factors()


@pytest.fixture(scope="function")
def example_conf(_example_conf):
    return _example_conf[:]


@pytest.fixture(scope="function")
def example_box(_example_system):
    _, _, box = _example_system
    return np.asarray(box / box.unit)
