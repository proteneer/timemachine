import numpy as np
import pytest

from timemachine.lib.potentials import NonbondedInteractionGroup


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
