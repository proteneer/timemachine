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
from timemachine.potentials import nonbonded


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(2022)
    yield


@pytest.fixture()
def rng():
    return np.random.default_rng(2022)


@pytest.fixture
def example_system():
    pdb_path = "tests/data/5dfr_solv_equil.pdb"
    host_pdb = app.PDBFile(pdb_path)
    ff = app.ForceField("amber99sbildn.xml", "tip3p.xml")
    return (
        ff.createSystem(host_pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False),
        host_pdb.positions,
        host_pdb.topology.getPeriodicBoxVectors(),
    )


@pytest.fixture
def example_nonbonded_params(example_system):
    host_system, _, _ = example_system
    host_fns, _ = openmm_deserializer.deserialize_system(host_system, cutoff=1.0)

    nonbonded_fn = None
    for f in host_fns:
        if isinstance(f, potentials.Nonbonded):
            nonbonded_fn = f

    assert nonbonded_fn is not None
    return nonbonded_fn.params


@pytest.fixture
def example_coords(example_system):
    _, host_coords, _ = example_system
    return np.array([[to_md_units(x), to_md_units(y), to_md_units(z)] for x, y, z in host_coords])


@pytest.fixture
def example_box(example_system):
    _, _, box = example_system
    return np.asarray(box / box.unit)


def test_nonbonded_interaction_group_zero_interactions(rng: np.random.Generator):
    num_atoms = 33
    num_row_atoms = 15
    beta = 2.0
    lamb = 0.1
    cutoff = 1.1
    box = 10.0 * np.eye(3)
    coords = rng.uniform(0, 1, size=(num_atoms, 3))
    row_atom_idxs = rng.choice(num_atoms, size=num_row_atoms, replace=False).astype(np.int32)

    # shift row atoms in x by twice the cutoff
    coords[row_atom_idxs, 0] += 2 * cutoff

    params = rng.uniform(0, 1, size=(num_atoms, 3))

    potential = NonbondedInteractionGroup(
        row_atom_idxs,
        np.zeros(num_atoms, dtype=np.int32),
        np.zeros(num_atoms, dtype=np.int32),
        beta,
        cutoff,
    )

    du_dx, du_dp, du_dl, u = potential.unbound_impl(np.float64).execute(coords, params, box, lamb)

    assert (du_dx == 0).all()
    assert (du_dp == 0).all()
    assert du_dl == 0
    assert u == 0


@pytest.mark.parametrize("lamb", [0.0, 0.1])
@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
@pytest.mark.parametrize("num_row_atoms", [1, 15])
@pytest.mark.parametrize("num_atoms", [33, 231, 1050])
def test_nonbonded_interaction_group_consistency_allpairs(
    num_atoms,
    num_row_atoms,
    precision,
    rtol,
    atol,
    cutoff,
    beta,
    lamb,
    example_nonbonded_params,
    example_coords,
    example_box,
    rng: np.random.Generator,
):
    """Compares with reference nonbonded_v3 potential, which computes
    the sum of all pairwise interactions. This uses the identity

      U = U_A + U_B + U_AB

    where
    - `U` is the all-pairs potential over all atoms
    - `U_A`, `U_B` are all-pairs potentials for interacting groups A
      and B, respectively
    - `U_AB` is the "interaction group" potential, i.e. the sum of
      pairwise interactions `(a, b)` where `a` is in `A` and `b` is in
      `B`

    The quantity `U` is computed using the reference potential over
    all atoms, and `U_A + U_B` computed using the reference potential
    over all atoms separated into 2 lambda planes according to which
    interacting group they belong
    """

    coords = example_coords[:num_atoms]
    params = example_nonbonded_params[:num_atoms, :]
    lambda_offset_idxs = np.zeros(num_atoms, dtype=np.int32)

    def make_reference_nonbonded(lambda_plane_idxs):
        return prepare_reference_nonbonded(
            params=params,
            exclusion_idxs=np.array([], dtype=np.int32),
            scales=np.zeros((0, 2), dtype=np.float64),
            lambda_plane_idxs=lambda_plane_idxs,
            lambda_offset_idxs=lambda_offset_idxs,
            beta=beta,
            cutoff=cutoff,
        )

    ref_allpairs = make_reference_nonbonded(np.zeros(num_atoms, dtype=np.int32))

    row_atom_idxs = rng.choice(num_atoms, size=num_row_atoms, replace=False).astype(np.int32)
    lambda_plane_idxs = np.zeros(num_atoms, dtype=np.int32)
    lambda_plane_idxs[row_atom_idxs] = 1

    ref_allpairs_minus_ixngroups = make_reference_nonbonded(lambda_plane_idxs)

    def ref_ixngroups(*args):
        return ref_allpairs(*args) - ref_allpairs_minus_ixngroups(*args)

    test_ixngroups = NonbondedInteractionGroup(
        row_atom_idxs,
        np.zeros(num_atoms, dtype=np.int32),  # lambda plane indices
        lambda_offset_idxs,
        beta,
        cutoff,
    )

    GradientTest().compare_forces(
        coords,
        params,
        example_box,
        lamb=lamb,
        ref_potential=ref_ixngroups,
        test_potential=test_ixngroups,
        rtol=rtol,
        atol=atol,
        precision=precision,
    )


@pytest.mark.parametrize("lamb", [0.0, 0.1])
@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
@pytest.mark.parametrize("num_row_atoms", [1, 15])
@pytest.mark.parametrize("num_atoms", [33, 231])
def test_nonbonded_interaction_group_correctness(
    num_atoms,
    num_row_atoms,
    precision,
    rtol,
    atol,
    cutoff,
    beta,
    lamb,
    example_nonbonded_params,
    example_coords,
    example_box,
    rng,
):
    "Compares with jax reference implementation."

    coords = example_coords[:num_atoms]
    params = example_nonbonded_params[:num_atoms, :]

    row_atom_idxs = rng.choice(num_atoms, size=num_row_atoms, replace=False).astype(np.int32)
    col_atom_idxs = np.setdiff1d(np.arange(num_atoms), row_atom_idxs)

    # hydrogen atoms have lennard-jones parameters set to 0
    atom_is_hydrogen = (params[:, 1] == 0) & (params[:, 2] == 0)

    def ref_ixngroups(coords, params, box, _):
        vdW, electrostatics, pairs = nonbonded.nonbonded_v3_interaction_groups(
            coords, params, box, row_atom_idxs, col_atom_idxs, beta, cutoff
        )

        # custom ops implementation returns du/dp = 0 for lennard-jones params at zero
        pair_has_hydrogen = atom_is_hydrogen[pairs[:, 0]] | atom_is_hydrogen[pairs[:, 1]]
        vdW_pinned = vdW.at[pair_has_hydrogen].set(0.0)

        total_energy = jax.numpy.sum(vdW_pinned + electrostatics)
        return total_energy

    test_ixngroups = NonbondedInteractionGroup(
        row_atom_idxs,
        np.zeros(num_atoms, dtype=np.int32),
        np.zeros(num_atoms, dtype=np.int32),
        beta,
        cutoff,
    )

    GradientTest().compare_forces(
        coords,
        params,
        example_box,
        lamb=lamb,
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
