import numpy as np
from jax import config, vmap

config.update("jax_enable_x64", True)

from functools import partial

from timemachine.maps.terminal_bonds import Interval, interval_map


def test_invertibility_of_interval_maps():
    """Test that we can construct invertible maps between intervals on the real line"""

    # construct a bunch of random intervals
    np.random.seed(2022)
    states = [Interval(np.random.rand(), 1 + np.random.rand()) for _ in range(50)]
    for state in states:
        state.validate()

    # generate test points up to within eps of interval bounds,
    # (eps slightly > 0 to avoid spurious <=, >= assertion errors near bounds...)
    eps = 1e-8

    def construct_map(src: Interval, dst: Interval):
        return partial(interval_map, src_lb=src.lower, src_ub=src.upper, dst_lb=dst.lower, dst_ub=dst.upper)

    # for each pair of states, compute f, f_inv on a bunch of points, assert self-consistency
    for src in states:
        for dst in states:
            f = construct_map(src, dst)
            f_inv = construct_map(dst, src)

            # x in src
            xs = np.linspace(src.lower + eps, src.upper - eps, 1000)
            np.testing.assert_array_less(src.lower, xs, err_msg="x not in support of src!")
            np.testing.assert_array_less(xs, src.upper, err_msg="x not in support of src!")

            # y=f(x) in dst
            ys = vmap(f)(xs)
            assert ys.shape == xs.shape
            np.testing.assert_array_less(dst.lower, ys, err_msg="y not in support of dst!")
            np.testing.assert_array_less(ys, dst.upper, err_msg="y not in support of dst!")

            # x_=f_inv(f(x))
            xs_ = vmap(f_inv)(ys)
            np.testing.assert_array_less(src.lower, xs_, err_msg="f_inv(f(x)) not in support of src!")
            np.testing.assert_array_less(xs_, src.upper, err_msg="f_inv(f(x)) not in support of src!")
            np.testing.assert_allclose(xs_, xs, err_msg="f_inv(f(x)) != x!")
