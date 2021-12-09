import pytest

import timemachine.lib.potentials as ps


def test_summed_potential_raises_on_inconsistent_lengths():

    with pytest.raises(ValueError) as excinfo:
        ps.SummedPotential([ps.HarmonicBond()], [])

    assert str(excinfo.value) == "number of potentials != number of parameter arrays"
