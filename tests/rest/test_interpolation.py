import re

import pytest

from timemachine.fe.rest.interpolation import get_interpolation_fxn


@pytest.mark.parametrize("interpolation", ["linear", "quadratic", "exponential"])
def test_raises_when_arg_outside_of_domain(interpolation):
    f = get_interpolation_fxn(interpolation, 1.0, 2.0)
    with pytest.raises(ValueError, match=re.escape("argument must be in [0, 1]")):
        _ = f(1.1)
    with pytest.raises(ValueError, match=re.escape("argument must be in [0, 1]")):
        _ = f(-0.1)
