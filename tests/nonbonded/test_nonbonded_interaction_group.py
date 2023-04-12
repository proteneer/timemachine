import jax.numpy as jnp
import numpy as np
import pytest
from common import GradientTest, gen_nonbonded_params_with_4d_offsets

from timemachine.potentials import Nonbonded, NonbondedInteractionGroup

pytestmark = [pytest.mark.memcheck]


def test_nonbonded_interaction_group_invalid_indices():
    with pytest.raises(RuntimeError, match="row_atom_idxs must be nonempty"):
        NonbondedInteractionGroup(1, [], 1.0, 1.0).to_gpu(np.float64).unbound_impl

    with pytest.raises(RuntimeError, match="atom indices must be unique"):
        NonbondedInteractionGroup(3, [1, 1], 1.0, 1.0).to_gpu(np.float64).unbound_impl

    with pytest.raises(RuntimeError, match="index values must be greater or equal to zero"):
        NonbondedInteractionGroup(3, [1, -1], 1.0, 1.0).to_gpu(np.float64).unbound_impl

    with pytest.raises(RuntimeError, match="index values must be less than N"):
        NonbondedInteractionGroup(3, [1, 100], 1.0, 1.0).to_gpu(np.float64).unbound_impl

    with pytest.raises(RuntimeError) as e:
        NonbondedInteractionGroup(3, [0, 1, 2], 1.0, 1.0).to_gpu(np.float64).unbound_impl
    assert "must be less then N(3) indices" == str(e.value)


def test_nonbonded_interaction_group_zero_interactions(rng: np.random.Generator):
    num_atoms = 33
    num_atoms_ligand = 15
    beta = 2.0
    cutoff = 1.1
    box = 10.0 * np.eye(3)
    conf = rng.uniform(0, 1, size=(num_atoms, 3))
    ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False).astype(np.int32)

    # shift ligand atoms in x by twice the cutoff
    conf[ligand_idxs, 0] += 2 * cutoff

    params = rng.uniform(0, 1, size=(num_atoms, 4))

    potential = NonbondedInteractionGroup(num_atoms, ligand_idxs, beta, cutoff)

    du_dx, du_dp, u = potential.to_gpu(np.float64).unbound_impl.execute(conf, params, box)

    assert (du_dx == 0).all()
    assert (du_dp == 0).all()
    assert u == 0


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
    example_nonbonded_potential,
    example_conf,
    example_box,
    rng,
):
    "Compares with jax reference implementation."

    conf = example_conf[:num_atoms]
    params = example_nonbonded_potential.params[:num_atoms, :]

    ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False).astype(np.int32)

    potential = NonbondedInteractionGroup(num_atoms, ligand_idxs, beta, cutoff)

    test_impl = potential.to_gpu(precision)

    for params in gen_nonbonded_params_with_4d_offsets(rng, params, cutoff):
        GradientTest().compare_forces(conf, params, example_box, potential, test_impl, rtol=rtol, atol=atol)
        GradientTest().assert_differentiable_interface_consistency(conf, params, example_box, test_impl)


@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
@pytest.mark.parametrize("num_atoms_ligand", [1, 15])
@pytest.mark.parametrize("num_atoms", [33, 231, 1050])
def test_nonbonded_interaction_group_consistency_allpairs_4d_decoupled(
    num_atoms,
    num_atoms_ligand,
    precision,
    rtol,
    atol,
    cutoff,
    beta,
    example_nonbonded_potential,
    example_conf,
    example_box,
    rng: np.random.Generator,
):
    """Compares with reference nonbonded potential, which computes the sum of
    all pairwise interactions. This uses the identity

      U = U_A + U_B + U_AB

    where
    * U is the all-pairs potential over all atoms
    * U_A, U_B are all-pairs potentials for interacting groups A and
      B, respectively
    * U_AB is the "interaction group" potential, i.e. the sum of
      pairwise interactions (a, b) where "a" is in A and "b" is in B

    * U is computed using the reference potential over all atoms
    * U_A + U_B is computed using the reference potential over all atoms,
      separated into 2 noninteracting groups in the 4th dimension
    """

    conf = example_conf[:num_atoms]
    params = example_nonbonded_potential.params[:num_atoms, :]

    ref_allpairs = Nonbonded(
        num_atoms,
        exclusion_idxs=np.array([], dtype=np.int32),
        scale_factors=np.zeros((0, 2), dtype=np.float64),
        beta=beta,
        cutoff=cutoff,
    )

    ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False).astype(np.int32)

    def make_ref_ixngroups():
        w_offsets = np.zeros(num_atoms)

        # ensure minimum distance between a host and ligand atom is >= cutoff
        # i.e. (w - cutoff) - cutoff > cutoff => w > 3 * cutoff
        w_offsets[ligand_idxs] = 3.01 * cutoff

        def ref_ixngroups(coords, params, box):
            U = ref_allpairs(coords, params, box)
            UA_plus_UB = ref_allpairs(coords, jnp.asarray(params).at[:, 3].add(w_offsets), box)
            return U - UA_plus_UB

        return ref_ixngroups

    ref_ixngroups = make_ref_ixngroups()
    test_ixngroups = NonbondedInteractionGroup(num_atoms, ligand_idxs, beta, cutoff).to_gpu(precision)

    for params in gen_nonbonded_params_with_4d_offsets(rng, params, cutoff):
        GradientTest().compare_forces(
            conf,
            params,
            example_box,
            ref_potential=ref_ixngroups,
            test_potential=test_ixngroups,
            rtol=rtol,
            atol=atol,
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
    example_nonbonded_potential,
    example_conf,
    example_box,
    rng: np.random.Generator,
):
    """Compares with reference nonbonded potential, which computes the sum of
    all pairwise interactions. This uses the identity

      U(x') - U(x) = U_AB(x') - U_AB(x)

    where
    * U is the all-pairs potential over all atoms
    * U_A, U_B are all-pairs potentials for interacting groups A and
      B, respectively
    * U_AB is the "interaction group" potential, i.e. the sum of
      pairwise interactions (a, b) where "a" is in A and "b" is in B
    * the transformation x -> x' does not affect U_A or U_B (e.g. a
      constant translation applied to each atom in one group)
    """

    conf = example_conf[:num_atoms]
    params = example_nonbonded_potential.params[:num_atoms, :]

    def ref_allpairs(conf):
        U_ref = Nonbonded(
            num_atoms,
            exclusion_idxs=np.array([], dtype=np.int32),
            scale_factors=np.zeros((0, 2), dtype=np.float64),
            beta=beta,
            cutoff=cutoff,
        )

        return U_ref(conf, params, example_box)

    ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False).astype(np.int32)

    test_impl = NonbondedInteractionGroup(num_atoms, ligand_idxs, beta, cutoff).to_gpu(precision).unbound_impl

    def test_ixngroups(conf):
        _, _, u = test_impl.execute(conf, params, example_box)
        return u

    conf_prime = np.array(conf)
    conf_prime[ligand_idxs] += rng.normal(0, 0.01, size=(3,))

    for params in gen_nonbonded_params_with_4d_offsets(rng, params, cutoff):
        ref_delta = ref_allpairs(conf_prime) - ref_allpairs(conf)
        test_delta = test_ixngroups(conf_prime) - test_ixngroups(conf)
        np.testing.assert_allclose(ref_delta, test_delta, rtol=rtol, atol=atol)


@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("num_atoms_ligand", [1, 15])
@pytest.mark.parametrize("num_atoms", [33, 231])
def test_nonbonded_interaction_group_set_atom_idxs(
    num_atoms, num_atoms_ligand, precision, cutoff, beta, rng: np.random.Generator
):
    box = 3.0 * np.eye(3)
    conf = rng.uniform(0, cutoff * 10, size=(num_atoms, 3))
    params = rng.uniform(0, 1, size=(num_atoms, 4))

    ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False).astype(np.int32)
    other_idxs = np.setdiff1d(np.arange(num_atoms), ligand_idxs)

    # Pick a subset to compare against, should produce different values
    secondary_ligand_set = rng.choice(other_idxs, size=(1), replace=False).astype(np.int32)

    potential = NonbondedInteractionGroup(num_atoms, ligand_idxs, beta, cutoff)
    unbound_pot = potential.to_gpu(precision).unbound_impl

    ref_du_dx, ref_du_dp, ref_u = unbound_pot.execute(
        conf,
        params,
        box,
    )

    # Set to first particle not in ligand_idxs, should produce different values
    unbound_pot.set_atom_idxs(secondary_ligand_set)

    diff_du_dx, diff_du_dp, diff_u = unbound_pot.execute(
        conf,
        params,
        box,
    )
    assert np.any(diff_du_dx != ref_du_dx)
    assert np.any(diff_du_dp != ref_du_dp)
    assert not np.allclose(ref_u, diff_u)

    # Reconstructing an Ixn group with the other set of atoms should be identical.
    potential2 = NonbondedInteractionGroup(num_atoms, secondary_ligand_set, beta, cutoff)
    unbound_pot2 = potential2.to_gpu(precision).unbound_impl

    diff_ref_du_dx, diff_ref_du_dp, diff_ref_u = unbound_pot2.execute(
        conf,
        params,
        box,
    )
    np.testing.assert_array_equal(diff_ref_du_dx, diff_du_dx)
    np.testing.assert_array_equal(diff_ref_du_dp, diff_du_dp)
    np.testing.assert_equal(diff_ref_u, diff_u)

    # Set back to the indices, but shuffled, should be identical to reference
    rng.shuffle(ligand_idxs)
    unbound_pot.set_atom_idxs(ligand_idxs)

    test_du_dx, test_du_dp, test_u = unbound_pot.execute(
        conf,
        params,
        box,
    )
    np.testing.assert_array_equal(test_du_dx, ref_du_dx)
    np.testing.assert_array_equal(test_du_dp, ref_du_dp)
    np.testing.assert_equal(test_u, ref_u)
