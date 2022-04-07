import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
from common import GradientTest, prepare_reference_nonbonded
from parameter_interpolation import gen_params

from timemachine.lib.potentials import NonbondedInteractionGroup, NonbondedInteractionGroupInterpolated
from timemachine.potentials import jax_utils, nonbonded

pytestmark = [pytest.mark.memcheck]


def test_nonbonded_interaction_group_invalid_indices():
    def make_potential(ligand_idxs, num_atoms):
        lambda_plane_idxs = [0] * num_atoms
        lambda_offset_idxs = [0] * num_atoms
        return NonbondedInteractionGroup(ligand_idxs, lambda_plane_idxs, lambda_offset_idxs, 1.0, 1.0).unbound_impl(
            np.float64
        )

    with pytest.raises(RuntimeError) as e:
        make_potential([], 1)
    assert "row_atom_idxs must be nonempty" in str(e)

    with pytest.raises(RuntimeError) as e:
        make_potential([1, 1], 3)
    assert "atom indices must be unique" in str(e)


def test_nonbonded_interaction_group_zero_interactions(rng: np.random.Generator):
    num_atoms = 33
    num_atoms_ligand = 15
    beta = 2.0
    lamb = 0.1
    cutoff = 1.1
    box = 10.0 * np.eye(3)
    conf = rng.uniform(0, 1, size=(num_atoms, 3))
    ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False).astype(np.int32)

    # shift ligand atoms in x by twice the cutoff
    conf[ligand_idxs, 0] += 2 * cutoff

    params = rng.uniform(0, 1, size=(num_atoms, 3))

    potential = NonbondedInteractionGroup(
        ligand_idxs,
        np.zeros(num_atoms, dtype=np.int32),
        np.zeros(num_atoms, dtype=np.int32),
        beta,
        cutoff,
    )

    du_dx, du_dp, du_dl, u = potential.unbound_impl(np.float64).execute(conf, params, box, lamb)

    assert (du_dx == 0).all()
    assert (du_dp == 0).all()
    assert du_dl == 0
    assert u == 0


@pytest.mark.parametrize("lamb", [0.0, 0.1])
@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
@pytest.mark.parametrize("num_atoms_ligand", [1, 15])
@pytest.mark.parametrize("num_atoms", [33, 231])
def test_nonbonded_interaction_group_correctness(
    num_atoms,
    num_atoms_ligand,
    precision,
    rtol,
    atol,
    cutoff,
    beta,
    lamb,
    example_nonbonded_params,
    example_conf,
    example_box,
    rng,
):
    "Compares with jax reference implementation."

    conf = example_conf[:num_atoms]
    params = example_nonbonded_params[:num_atoms, :]

    lambda_plane_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)
    lambda_offset_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)

    ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False).astype(np.int32)
    host_idxs = np.setdiff1d(np.arange(num_atoms), ligand_idxs)

    def ref_ixngroups(conf, params, box, lamb):

        # compute 4d coordinates
        w = jax_utils.compute_lifting_parameter(lamb, lambda_plane_idxs, lambda_offset_idxs, cutoff)
        conf_4d = jax_utils.augment_dim(conf, w)
        box_4d = (1000 * jax.numpy.eye(4)).at[:3, :3].set(box)

        vdW, electrostatics = nonbonded.nonbonded_v3_interaction_groups(
            conf_4d, params, box_4d, ligand_idxs, host_idxs, beta, cutoff
        )
        return jax.numpy.sum(vdW + electrostatics)

    test_ixngroups = NonbondedInteractionGroup(
        ligand_idxs,
        lambda_plane_idxs,
        lambda_offset_idxs,
        beta,
        cutoff,
    )

    GradientTest().compare_forces(
        conf,
        params,
        example_box,
        [lamb],
        ref_potential=ref_ixngroups,
        test_potential=test_ixngroups,
        rtol=rtol,
        atol=atol,
        precision=precision,
    )


@pytest.mark.parametrize("lamb", [0.0, 0.1, 0.9, 1.0])
@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
@pytest.mark.parametrize("num_atoms_ligand", [1, 15])
@pytest.mark.parametrize("num_atoms", [33])
def test_nonbonded_interaction_group_interpolated_correctness(
    num_atoms,
    num_atoms_ligand,
    precision,
    rtol,
    atol,
    cutoff,
    beta,
    lamb,
    example_nonbonded_params,
    example_conf,
    example_box,
    rng,
):
    "Compares with jax reference implementation, with parameter interpolation."

    conf = example_conf[:num_atoms]
    params_initial = example_nonbonded_params[:num_atoms, :]
    params = gen_params(params_initial, rng)

    lambda_plane_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)
    lambda_offset_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)

    ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False).astype(np.int32)
    host_idxs = np.setdiff1d(np.arange(num_atoms), ligand_idxs)

    @nonbonded.interpolated
    def ref_ixngroups(conf, params, box, lamb):

        # compute 4d coordinates
        w = jax_utils.compute_lifting_parameter(lamb, lambda_plane_idxs, lambda_offset_idxs, cutoff)
        conf_4d = jax_utils.augment_dim(conf, w)
        box_4d = (1000 * jax.numpy.eye(4)).at[:3, :3].set(box)

        vdW, electrostatics = nonbonded.nonbonded_v3_interaction_groups(
            conf_4d, params, box_4d, ligand_idxs, host_idxs, beta, cutoff
        )
        return jax.numpy.sum(vdW + electrostatics)

    test_ixngroups = NonbondedInteractionGroupInterpolated(
        ligand_idxs,
        lambda_plane_idxs,
        lambda_offset_idxs,
        beta,
        cutoff,
    )

    GradientTest().compare_forces(
        conf,
        params,
        example_box,
        [lamb],
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
@pytest.mark.parametrize("num_atoms_ligand", [1, 15])
@pytest.mark.parametrize("num_atoms", [33, 231, 1050])
def test_nonbonded_interaction_group_consistency_allpairs_lambda_planes(
    num_atoms,
    num_atoms_ligand,
    precision,
    rtol,
    atol,
    cutoff,
    beta,
    lamb,
    example_nonbonded_params,
    example_conf,
    example_box,
    rng: np.random.Generator,
):
    """Compares with reference nonbonded_v3 potential, which computes
    the sum of all pairwise interactions. This uses the identity

      U = U_A + U_B + U_AB

    where
    - U is the all-pairs potential over all atoms
    - U_A, U_B are all-pairs potentials for interacting groups A and
      B, respectively
    - U_AB is the "interaction group" potential, i.e. the sum of
      pairwise interactions (a, b) where "a" is in A and "b" is in B

    U is computed using the reference potential over all atoms, and
    U_A + U_B computed using the reference potential over all atoms
    separated into 2 lambda planes according to which interacting
    group they belong
    """

    conf = example_conf[:num_atoms]
    params = example_nonbonded_params[:num_atoms, :]

    max_abs_offset_idx = 2  # i.e., lambda_offset_idxs in {-2, -1, 0, 1, 2}
    lambda_offset_idxs = rng.integers(-max_abs_offset_idx, max_abs_offset_idx + 1, size=(num_atoms,), dtype=np.int32)

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

    ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False).astype(np.int32)

    # for reference U_A + U_B computation, ensure minimum distance
    # between a host and ligand atom is at least one cutoff distance
    # when lambda = 1
    lambda_plane_idxs = np.zeros(num_atoms, dtype=np.int32)
    lambda_plane_idxs[ligand_idxs] = 2 * max_abs_offset_idx + 1

    ref_allpairs_minus_ixngroups = make_reference_nonbonded(lambda_plane_idxs)

    def ref_ixngroups(*args):
        return ref_allpairs(*args) - ref_allpairs_minus_ixngroups(*args)

    test_ixngroups = NonbondedInteractionGroup(
        ligand_idxs,
        np.zeros(num_atoms, dtype=np.int32),  # lambda plane indices
        lambda_offset_idxs,
        beta,
        cutoff,
    )

    GradientTest().compare_forces(
        conf,
        params,
        example_box,
        [lamb],
        ref_potential=ref_ixngroups,
        test_potential=test_ixngroups,
        rtol=rtol,
        atol=atol,
        precision=precision,
    )


@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
@pytest.mark.parametrize("num_atoms_ligand", [1, 15])
@pytest.mark.parametrize("num_atoms", [33, 231])
def test_nonbonded_interaction_group_consistency_allpairs_constant_shift(
    num_atoms,
    num_atoms_ligand,
    precision,
    rtol,
    atol,
    cutoff,
    beta,
    example_nonbonded_params,
    example_conf,
    example_box,
    rng: np.random.Generator,
):
    """Compares with reference nonbonded_v3 potential, which computes
    the sum of all pairwise interactions. This uses the identity

      U(x') - U(x) = U_AB(x') - U_AB(x)

    where
    - U is the all-pairs potential over all atoms
    - U_A, U_B are all-pairs potentials for interacting groups A and
      B, respectively
    - U_AB is the "interaction group" potential, i.e. the sum of
      pairwise interactions (a, b) where "a" is in A and "b" is in B
    - the transformation x -> x' does not affect U_A or U_B (e.g. a
      constant translation applied to each atom in one group)
    """

    conf = example_conf[:num_atoms]
    params = example_nonbonded_params[:num_atoms, :]

    lambda_plane_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)
    lambda_offset_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)

    def ref_allpairs(conf, lamb):
        return prepare_reference_nonbonded(
            params=params,
            exclusion_idxs=np.array([], dtype=np.int32),
            scales=np.zeros((0, 2), dtype=np.float64),
            lambda_plane_idxs=lambda_plane_idxs,
            lambda_offset_idxs=lambda_offset_idxs,
            beta=beta,
            cutoff=cutoff,
        )(conf, params, example_box, lamb)

    ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False).astype(np.int32)

    test_impl = NonbondedInteractionGroup(
        ligand_idxs,
        lambda_plane_idxs,
        lambda_offset_idxs,
        beta,
        cutoff,
    ).unbound_impl(precision)

    def test_ixngroups(conf, lamb):
        _, _, _, u = test_impl.execute(conf, params, example_box, lamb)
        return u

    conf_prime = np.array(conf)
    conf_prime[ligand_idxs] += rng.normal(0, 0.01, size=(3,))

    for lam in [0.0, 0.1]:
        ref_delta = ref_allpairs(conf_prime, lam) - ref_allpairs(conf, lam)
        test_delta = test_ixngroups(conf_prime, lam) - test_ixngroups(conf, lam)
        np.testing.assert_allclose(ref_delta, test_delta, rtol=rtol, atol=atol)
