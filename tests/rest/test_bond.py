import pytest

from timemachine.fe.rest.bond import CanonicalAngle, CanonicalBond, CanonicalTorsion


def test_rejects_non_canonical():
    with pytest.raises(ValueError):
        _ = CanonicalBond(2, 1)
    with pytest.raises(ValueError):
        _ = CanonicalAngle(3, 2, 1)
    with pytest.raises(ValueError):
        _ = CanonicalTorsion(4, 3, 2, 1)


def test_eq():
    assert CanonicalBond(1, 2) == CanonicalBond.from_idxs(1, 2)
    assert CanonicalBond(1, 2) == CanonicalBond.from_idxs(2, 1)

    assert CanonicalAngle(1, 2, 3) == CanonicalAngle.from_idxs(1, 2, 3)
    assert CanonicalAngle(1, 2, 3) == CanonicalAngle.from_idxs(3, 2, 1)
    assert CanonicalAngle(1, 2, 3) != CanonicalAngle.from_idxs(2, 3, 1)
    assert CanonicalAngle(1, 2, 3) != CanonicalAngle.from_idxs(2, 1, 3)

    assert CanonicalTorsion(1, 2, 3, 4) == CanonicalTorsion.from_idxs(1, 2, 3, 4)
    assert CanonicalTorsion(1, 2, 3, 4) == CanonicalTorsion.from_idxs(4, 3, 2, 1)
    assert CanonicalTorsion(1, 2, 3, 4) != CanonicalTorsion.from_idxs(2, 3, 4, 1)
    assert CanonicalTorsion(1, 2, 3, 4) != CanonicalTorsion.from_idxs(4, 2, 3, 1)
