import numpy as np
import pytest

from timemachine.potentials import BoundPotential, HarmonicAngle, HarmonicBond, Potential
from timemachine.potentials.potential import get_bound_potential_by_type, get_potential_by_type

pytestmark = [pytest.mark.nogpu]


def test_get_potential_by_type():
    with pytest.raises(ValueError, match="Unable to find potential of type"):
        get_potential_by_type([], HarmonicBond)

    pots = [HarmonicAngle(idxs=np.array([[0, 1, 2]], dtype=np.int32))]
    with pytest.raises(ValueError, match="Unable to find potential of type"):
        get_potential_by_type(pots, HarmonicBond)

    pots.append(HarmonicBond(idxs=np.array([[0, 1]], dtype=np.int32)))
    bonded = get_potential_by_type(pots, HarmonicBond)
    assert isinstance(bonded, Potential)
    assert isinstance(bonded, HarmonicBond)


def test_get_bound_potential_by_type():
    with pytest.raises(ValueError, match="Unable to find potential of type"):
        get_bound_potential_by_type([], HarmonicBond)

    bps = [
        BoundPotential(
            potential=HarmonicAngle(idxs=np.array([[0, 1, 2]], dtype=np.int32)), params=np.array([[0.0, 0.0]])
        )
    ]
    with pytest.raises(ValueError, match="Unable to find potential of type"):
        get_bound_potential_by_type(bps, HarmonicBond)

    bps.append(
        BoundPotential(potential=HarmonicBond(idxs=np.array([[0, 1]], dtype=np.int32)), params=np.array([[0.0, 0.0]]))
    )

    bonded = get_bound_potential_by_type(bps, HarmonicBond)
    assert isinstance(bonded, BoundPotential)
    assert isinstance(bonded.potential, HarmonicBond)
