import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
from common import GradientTest, prepare_reference_nonbonded
from simtk.openmm import app

from timemachine.fe.utils import to_md_units
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import potentials
from timemachine.lib.potentials import NonbondedInteractionGroup


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(2021)
    yield


@pytest.fixture
def test_system():
    pdb_path = "tests/data/5dfr_solv_equil.pdb"
    host_pdb = app.PDBFile(pdb_path)
    ff = app.ForceField("amber99sbildn.xml", "tip3p.xml")
    return (
        ff.createSystem(host_pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False),
        host_pdb.positions,
        host_pdb.topology.getPeriodicBoxVectors(),
    )


@pytest.fixture
def ref_nonbonded_potential(test_system):
    host_system, _, _ = test_system
    host_fns, _ = openmm_deserializer.deserialize_system(host_system, cutoff=1.0)

    nonbonded_fn = None
    for f in host_fns:
        if isinstance(f, potentials.Nonbonded):
            nonbonded_fn = f

    return nonbonded_fn


@pytest.fixture
def test_conf(test_system):
    _, host_coords, _ = test_system
    return np.array([[to_md_units(x), to_md_units(y), to_md_units(z)] for x, y, z in host_coords])


@pytest.fixture
def test_box(test_system):
    _, _, box = test_system
    return np.asarray(box / box.unit)


@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
@pytest.mark.parametrize("num_row_atoms", [1, 15])
@pytest.mark.parametrize("num_atoms", [33, 231, 1050])
def test_nonbonded_interaction_group_correctness(
    num_atoms, num_row_atoms, precision, rtol, atol, cutoff, beta, ref_nonbonded_potential, test_conf, test_box
):

    test_conf = test_conf[:num_atoms]
    test_params = ref_nonbonded_potential.params[:num_atoms, :]
    test_lambda_offset_idxs = np.zeros(num_atoms, dtype=np.int32)

    def make_reference_nonbonded(lambda_plane_idxs):
        return prepare_reference_nonbonded(
            params=test_params,
            exclusion_idxs=np.array([], dtype=np.int32),
            scales=np.zeros((0, 2), dtype=np.float64),
            lambda_plane_idxs=lambda_plane_idxs,
            lambda_offset_idxs=test_lambda_offset_idxs,
            beta=beta,
            cutoff=cutoff,
        )

    ref_allpairs = make_reference_nonbonded(np.zeros(num_atoms, dtype=np.int32))

    num_col_atoms = num_atoms - num_row_atoms

    ref_allpairs_minus_ixngroups = make_reference_nonbonded(
        np.concatenate((np.zeros(num_row_atoms, dtype=np.int32), np.ones(num_col_atoms, dtype=np.int32)))
    )

    def ref_ixngroups(*args):
        return ref_allpairs(*args) - ref_allpairs_minus_ixngroups(*args)

    test_ixngroups = NonbondedInteractionGroup(
        np.arange(0, num_row_atoms, dtype=np.int32),
        np.zeros(num_atoms, dtype=np.int32),  # lambda plane indices
        test_lambda_offset_idxs,
        beta,
        cutoff,
    )

    GradientTest().compare_forces(
        test_conf,
        test_params,
        test_box,
        lamb=0.1,
        ref_potential=ref_ixngroups,
        test_potential=test_ixngroups,
        rtol=rtol,
        atol=atol,
        precision=precision,
    )


def test_nonbonded_interaction_group_invalid_indices():
    def make_potential(row_atom_idxs, num_atoms):
        lambda_plane_idxs = [0] * num_atoms
        lambda_offset_idxs = [0] * num_atoms
        return NonbondedInteractionGroup(row_atom_idxs, lambda_plane_idxs, lambda_offset_idxs, 1.0, 1.0).unbound_impl(
            np.float64
        )

    with pytest.raises(RuntimeError) as e:
        make_potential([], 1)
    assert "row_atom_idxs must be nonempty" in str(e)

    with pytest.raises(RuntimeError) as e:
        make_potential([1, 1], 3)
    assert "atom indices must be unique" in str(e)
