from collections.abc import Iterable

import numpy as np
import pytest
from common import gen_nonbonded_params_with_4d_offsets

from timemachine.potentials import (
    FanoutSummedPotential,
    Nonbonded,
    NonbondedAllPairs,
    NonbondedExclusions,
    NonbondedInteractionGroup,
)

pytestmark = [pytest.mark.memcheck]


def filter_valid_exclusions(
    num_atoms: int, exclusions: Iterable[tuple[int, int]], scales: Iterable[tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    filtered_pairs = (((i, j), scales) for (i, j), scales in zip(exclusions, scales) if i < num_atoms and j < num_atoms)
    idxs, scales = zip(*filtered_pairs)
    return np.array(idxs, dtype=np.int32), np.array(scales)


@pytest.mark.parametrize("disable_hilbert_sort", [False, True])
@pytest.mark.parametrize("beta", [1.0, 2.0])
@pytest.mark.parametrize("cutoff", [0.7, 1.1])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize(
    "num_atoms,num_atoms_ligand,num_ixn_groups", [(33, 1, 0), (33, 1, 1), (231, 15, 2), (4080, 1050, 5)]
)
def test_nonbonded_consistency(
    num_atoms_ligand,
    num_atoms,
    num_ixn_groups,
    precision,
    cutoff,
    beta,
    disable_hilbert_sort,
    example_nonbonded_potential,
    example_conf,
    example_box,
    rng: np.random.Generator,
):
    conf = example_conf[:num_atoms, :]
    example_bp = example_nonbonded_potential
    params = example_nonbonded_potential.params[:num_atoms, :]
    exclusion_idxs, exclusion_scales = filter_valid_exclusions(
        num_atoms, example_bp.potential.exclusion_idxs, example_bp.potential.scale_factors
    )

    ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False).astype(np.int32)
    host_idxs = np.setdiff1d(np.arange(num_atoms), ligand_idxs).astype(np.int32)

    # Partition host_idxs into num_ixn_groups
    if num_ixn_groups:
        group_idxs = rng.choice(num_ixn_groups, size=(len(host_idxs),), replace=True).astype(np.int32)
        all_col_atom_idxs = [host_idxs[group_idxs == i] for i in range(num_ixn_groups)]
    else:
        # Test case where col_atom_idxs is None
        all_col_atom_idxs = [None]

    ref_impl = (
        Nonbonded(num_atoms, exclusion_idxs, exclusion_scales, beta, cutoff, None, disable_hilbert_sort)
        .to_gpu(precision)
        .unbound_impl
    )

    def make_allpairs_potential(atom_idxs):
        return NonbondedAllPairs(num_atoms, beta, cutoff, atom_idxs, disable_hilbert_sort)

    def make_ixngroup_potential(ligand_idxs, col_atom_idxs):
        return NonbondedInteractionGroup(num_atoms, ligand_idxs, beta, cutoff, col_atom_idxs, disable_hilbert_sort)

    def make_exclusions_potential(exclusion_idxs, exclusion_scales):
        return NonbondedExclusions(exclusion_idxs, exclusion_scales, beta, cutoff)

    test_impl = (
        FanoutSummedPotential(
            [
                make_allpairs_potential(host_idxs),
                make_allpairs_potential(ligand_idxs),
                make_exclusions_potential(exclusion_idxs, exclusion_scales),
            ]
            + [make_ixngroup_potential(ligand_idxs, col_atom_idxs) for col_atom_idxs in all_col_atom_idxs]
        )
        .to_gpu(precision)
        .unbound_impl
    )

    for params_ in gen_nonbonded_params_with_4d_offsets(rng, params, cutoff):
        du_dx_ref, du_dp_ref, u_ref = ref_impl.execute(conf, params_, example_box)
        du_dx_test, du_dp_test, u_test = test_impl.execute(conf, params_, example_box)

        np.testing.assert_array_equal(du_dx_test, du_dx_ref)
        np.testing.assert_array_equal(du_dp_test, du_dp_ref)
        assert u_test == u_ref
