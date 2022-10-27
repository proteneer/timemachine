from typing import Iterable, Tuple

import numpy as np
import pytest
from common import gen_nonbonded_params_with_4d_offsets

from timemachine.lib.potentials import (
    FanoutSummedPotential,
    Nonbonded,
    NonbondedAllPairs,
    NonbondedInteractionGroup,
    NonbondedPairListNegated,
)


def filter_valid_exclusions(
    num_atoms: int, exclusions: Iterable[Tuple[int, int]], scales: Iterable[Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    filtered_pairs = (((i, j), scales) for (i, j), scales in zip(exclusions, scales) if i < num_atoms and j < num_atoms)
    idxs, scales = zip(*filtered_pairs)
    return np.array(idxs, dtype=np.int32), np.array(scales)


@pytest.mark.parametrize("beta", [1.0, 2.0])
@pytest.mark.parametrize("cutoff", [0.7, 1.1])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("num_atoms,num_atoms_ligand", [(33, 1), (231, 15), (4080, 1050)])
def test_nonbonded_consistency(
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
    params = example_nonbonded_potential.params[:num_atoms, :]

    exclusion_idxs, exclusion_scales = filter_valid_exclusions(
        num_atoms,
        example_nonbonded_potential.get_exclusion_idxs(),
        example_nonbonded_potential.get_scale_factors(),
    )

    ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False).astype(np.int32)
    host_idxs = np.setdiff1d(np.arange(num_atoms), ligand_idxs).astype(np.int32)

    ref_impl = Nonbonded(num_atoms, exclusion_idxs, exclusion_scales, beta, cutoff).unbound_impl(precision)

    def make_allpairs_potential(atom_idxs):
        return NonbondedAllPairs(num_atoms, beta, cutoff, atom_idxs)

    def make_ixngroup_potential(ligand_idxs):
        return NonbondedInteractionGroup(num_atoms, ligand_idxs, beta, cutoff)

    def make_pairlist_potential(exclusion_idxs, exclusion_scales):
        return NonbondedPairListNegated(exclusion_idxs, exclusion_scales, beta, cutoff)

    test_impl = FanoutSummedPotential(
        [
            make_allpairs_potential(host_idxs),
            make_allpairs_potential(ligand_idxs),
            make_ixngroup_potential(ligand_idxs),
            make_pairlist_potential(exclusion_idxs, exclusion_scales),
        ]
    ).unbound_impl(precision)

    def test():
        for params_ in gen_nonbonded_params_with_4d_offsets(rng, params, cutoff):
            du_dx_ref, du_dp_ref, du_dl_ref, u_ref = ref_impl.execute(conf, params_, example_box, 0.0)
            du_dx_test, du_dp_test, du_dl_test, u_test = test_impl.execute(conf, params_, example_box, 0.0)

            np.testing.assert_array_equal(du_dx_test, du_dx_ref)
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
