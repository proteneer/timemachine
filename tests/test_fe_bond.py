import pytest

from timemachine.fe.bond import CanonicalAngle, CanonicalBond, CanonicalTorsion


def test_rejects_non_canonical():
    with pytest.raises(ValueError):
        _ = CanonicalBond(2, 1)
    with pytest.raises(ValueError):
        _ = CanonicalAngle(3, 2, 1)
    with pytest.raises(ValueError):
        _ = CanonicalTorsion(4, 3, 2, 1)


def test_eq():
    assert CanonicalBond(1, 2) == CanonicalBond.from_idxs(1, 2)
