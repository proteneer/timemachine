from typing import Iterable, Tuple

import numpy as np
import pytest
from parameter_interpolation import gen_params

from timemachine.lib.potentials import (
    FanoutSummedPotential,
    Nonbonded,
    NonbondedAllPairs,
    NonbondedAllPairsInterpolated,
    NonbondedInteractionGroup,
    NonbondedInteractionGroupInterpolated,
    NonbondedInterpolated,
    NonbondedPairListNegated,
    NonbondedPairListNegatedInterpolated,
)


def filter_valid_exclusions(
    num_atoms: int, exclusions: Iterable[Tuple[int, int]], scales: Iterable[Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    filtered_pairs = (((i, j), scales) for (i, j), scales in zip(exclusions, scales) if i < num_atoms and j < num_atoms)
    idxs, scales = zip(*filtered_pairs)
    return np.array(idxs, dtype=np.int32), np.array(scales)


@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("num_atoms,num_atoms_ligand", [(33, 1), (4080, 1050)])
@pytest.mark.parametrize("interpolated", [False, True])
def test_nonbonded_consistency(
    interpolated,
    num_atoms_ligand,
    num_atoms,
    precision,
    cutoff,
    beta,
    example_nonbonded_potential,
    example_conf,
    example_box,
    rng: np.random.Generator,
):
    conf = example_conf[:num_atoms, :]
    params_initial = example_nonbonded_potential.params[:num_atoms, :]
    params = gen_params(params_initial, rng) if interpolated else params_initial

    exclusion_idxs, exclusion_scales = filter_valid_exclusions(
        num_atoms,
        example_nonbonded_potential.get_exclusion_idxs(),
        example_nonbonded_potential.get_scale_factors(),
    )

    lambda_plane_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)
    lambda_offset_idxs = rng.integers(-2, 3, size=(num_atoms,), dtype=np.int32)

    ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False).astype(np.uint32)
    host_idxs = np.setdiff1d(np.arange(num_atoms), ligand_idxs).astype(np.uint32)

    make_ref_potential = NonbondedInterpolated if interpolated else Nonbonded
    ref_impl = make_ref_potential(
        exclusion_idxs, exclusion_scales, lambda_plane_idxs, lambda_offset_idxs, beta, cutoff
    ).unbound_impl(precision)

    def make_allpairs_potential(atom_idxs):
        make_potential = NonbondedAllPairsInterpolated if interpolated else NonbondedAllPairs
        return make_potential(
            lambda_plane_idxs,
            lambda_offset_idxs,
            beta,
            cutoff,
            atom_idxs,
        )

    def make_ixngroup_potential(ligand_idxs):
        make_potential = NonbondedInteractionGroupInterpolated if interpolated else NonbondedInteractionGroup
        return make_potential(
            ligand_idxs,
            lambda_plane_idxs,
            lambda_offset_idxs,
            beta,
            cutoff,
        )

    def make_pairlist_potential(exclusion_idxs, exclusion_scales):
        make_potential = NonbondedPairListNegatedInterpolated if interpolated else NonbondedPairListNegated
        return make_potential(exclusion_idxs, exclusion_scales, lambda_plane_idxs, lambda_offset_idxs, beta, cutoff)

    test_impl = FanoutSummedPotential(
        [
            make_allpairs_potential(host_idxs),
            make_allpairs_potential(ligand_idxs),
            make_ixngroup_potential(ligand_idxs.astype(np.int32)),
            make_pairlist_potential(exclusion_idxs, exclusion_scales),
        ]
    ).unbound_impl(precision)

    def test():
        for lam in [0.0, 0.1]:
            du_dx_ref, du_dp_ref, du_dl_ref, u_ref = ref_impl.execute(conf, params, example_box, lam)
            du_dx_test, du_dp_test, du_dl_test, u_test = test_impl.execute(conf, params, example_box, lam)

            np.testing.assert_array_equal(du_dx_test, du_dx_ref)

            if interpolated:
                # NOTE: bitwise equivalence is not currently possible for the
                # interpolated case. To see this, note that the interpolated
                # energy is given by
                #
                #   u(p0, p1) = (1 - lam) * F(p0) + lam * F(p1)
                #
                # In particular,
                #
                #   du_dp1 = lam * f(p1)
                #
                # where f(p) = F'(p). The reference potential effectively sums
                # over interactions before multiplication by \lambda
                #
                #   du_dp1_ref = lam * fixed_sum(f_host(p1), f_ligand(p1), f_host_ligand(p1))
                #
                # while the test potential (because it's implemented using
                # SummedPotential), effectively distributes multiplication by
                # \lambda into the sum
                #
                #   du_dp1_test = fixed_sum(lam * f_host(p1), lam * f_ligand(p1), lam * f_host_ligand(p1))
                #
                # Since `c * fixed_sum(x, y)` is not bitwise equivalent to
                # `fixed_sum(c * x, c * y)` in general, the reference and test
                # du_dps are not guaranteed to be bitwise equivalent in the
                # interpolated case.
                np.testing.assert_allclose(du_dp_test, du_dp_ref, rtol=1e-8, atol=1e-8)
            else:
                np.testing.assert_array_equal(du_dp_test, du_dp_ref)

            np.testing.assert_array_equal(du_dl_test, du_dl_ref)
            assert u_test == u_ref

    test()

    # Test with hilbert sorting disabled
    ref_impl.disable_hilbert_sort()

    # NonbondedAllPairs and NonbondedInteractionGroup have a
    # disable_hilbert_sort method; NonbondedPairList doesn't
    for impl in test_impl.get_potentials():
        if hasattr(impl, "disable_hilbert_sort"):
            impl.disable_hilbert_sort()

    test()
