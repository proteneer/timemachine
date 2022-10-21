import jax.numpy as jnp
import numpy as np
import pytest

from tests.common import GradientTest
from tests.nonbonded import gen_params_with_4d_offsets
from timemachine.lib.potentials import NonbondedInteractionGroup
from timemachine.potentials import generic

pytestmark = [pytest.mark.memcheck]


def test_nonbonded_interaction_group_invalid_indices():
    with pytest.raises(RuntimeError) as e:
        NonbondedInteractionGroup(1, [], 1.0, 1.0).unbound_impl(np.float64)
    assert "row_atom_idxs must be nonempty" in str(e)

    with pytest.raises(RuntimeError) as e:
        NonbondedInteractionGroup(3, [1, 1], 1.0, 1.0).unbound_impl(np.float64)
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

    params = rng.uniform(0, 1, size=(num_atoms, 4))

    potential = NonbondedInteractionGroup(num_atoms, ligand_idxs, beta, cutoff)

    du_dx, du_dp, du_dl, u = potential.unbound_impl(np.float64).execute(conf, params, box, lamb)

    assert (du_dx == 0).all()
    assert (du_dp == 0).all()
    assert du_dl == 0
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

    potential = generic.NonbondedInteractionGroup(num_atoms, ligand_idxs, beta, cutoff)

    GradientTest().compare_forces_gpu_vs_reference(
        conf,
        gen_params_with_4d_offsets(rng, params, -2 * cutoff, 2 * cutoff, 3),
        example_box,
        potential,
        rtol=rtol,
        atol=atol,
        precision=precision,
    )


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
    * U_A + U_B is computed using the reference potential over all atoms
      separated into 2 noninteracting groups in the w coordinate
    """

    conf = example_conf[:num_atoms]
    params = example_nonbonded_potential.params[:num_atoms, :]

    ref_allpairs = generic.Nonbonded(
        num_atoms,
        exclusion_idxs=np.array([], dtype=np.int32),
        scale_factors=np.zeros((0, 2), dtype=np.float64),
        beta=beta,
        cutoff=cutoff,
    ).to_reference()

    ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False).astype(np.int32)

    def make_ref_ixngroups():
        w_offsets = np.zeros(num_atoms)

        # ensure minimum distance between a host and ligand atom is >= cutoff
        # i.e. (w - cutoff) - cutoff > cutoff => w > 3 * cutoff
        w_offsets[ligand_idxs] = 3.01 * cutoff

        def ref_ixngroups(coords, params, box, lam):
            U = ref_allpairs(coords, params, box, lam)
            UA_plus_UB = ref_allpairs(coords, jnp.asarray(params).at[:, 3].add(w_offsets), box, lam)
            return U - UA_plus_UB

        return ref_ixngroups

    ref_ixngroups = make_ref_ixngroups()
    test_ixngroups = NonbondedInteractionGroup(num_atoms, ligand_idxs, beta, cutoff)

    GradientTest().compare_forces(
        conf,
        gen_params_with_4d_offsets(rng, params, -cutoff, cutoff, 3),
        example_box,
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
    example_nonbonded_potential,
    example_conf,
    example_box,
    rng: np.random.Generator,
):
    """Compares with reference nonbonded potential, which computes
    the sum of all pairwise interactions. This uses the identity

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
        U_ref = generic.Nonbonded(
            num_atoms,
            exclusion_idxs=np.array([], dtype=np.int32),
            scale_factors=np.zeros((0, 2), dtype=np.float64),
            beta=beta,
            cutoff=cutoff,
        ).to_reference()

        return U_ref(conf, params, example_box, 0.0)

    ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False).astype(np.int32)

    test_impl = NonbondedInteractionGroup(num_atoms, ligand_idxs, beta, cutoff).unbound_impl(precision)

    def test_ixngroups(conf):
        _, _, _, u = test_impl.execute(conf, params, example_box, 0.0)
        return u

    conf_prime = np.array(conf)
    conf_prime[ligand_idxs] += rng.normal(0, 0.01, size=(3,))

    for params in gen_params_with_4d_offsets(rng, params, -2 * cutoff, 2 * cutoff, 3):
        ref_delta = ref_allpairs(conf_prime) - ref_allpairs(conf)
        test_delta = test_ixngroups(conf_prime) - test_ixngroups(conf)
        np.testing.assert_allclose(ref_delta, test_delta, rtol=rtol, atol=atol)
