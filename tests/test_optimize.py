import numpy as np
from common import get_110_ccc_ff
from jax import config, grad
from jax import numpy as jnp
from jax import tree_util

from timemachine.optimize.precondition import learning_rates_like_params
from timemachine.optimize.step import truncated_step
from timemachine.optimize.utils import flatten_and_unflatten

config.update("jax_enable_x64", True)


def check_flatten_and_unflatten_roundtrip(example_pytree):
    """assert that structures with array-valued leaves can be successfully
    flattened and unflattened"""

    flatten, unflatten = flatten_and_unflatten(example_pytree)

    original_vector = flatten(example_pytree)
    roundtripped_vector = flatten(unflatten(original_vector))

    assert (original_vector == roundtripped_vector).all()

    original_structure = tree_util.tree_structure(example_pytree)
    roundtripped_structure = tree_util.tree_structure(unflatten(flatten(example_pytree)))

    assert original_structure == roundtripped_structure


def test_flatten_and_unflatten_dict():
    """flatten/unflatten a nested object with array-valued leaves"""

    example_pytree = dict(
        Bonds=np.random.randn(50, 2),
        Angles=np.random.randn(60, 3),
        Extras=dict(
            GlobalVars=[
                jnp.arange(70),
                np.random.randn(80, 4),
            ]
        ),
        Bias=1.2,
        Multiplier=np.array(2.5),
    )

    check_flatten_and_unflatten_roundtrip(example_pytree)


def test_flatten_and_unflatten_ordered_params():
    """flatten/unflatten a Forcefield(ff_handlers).get_ordered_params()"""

    forcefield = get_110_ccc_ff()
    ordered_params = forcefield.get_ordered_params()

    check_flatten_and_unflatten_roundtrip(ordered_params)


def test_learning_rates_like_params():
    """assert shape compatibility btwn ordered params and ordered learning rates"""

    forcefield = get_110_ccc_ff()
    ordered_handles = forcefield.get_ordered_handles()
    ordered_params = forcefield.get_ordered_params()

    ordered_lr = learning_rates_like_params(ordered_handles, ordered_params)

    for (p, l) in zip(ordered_params, ordered_lr):
        assert p.shape == l.shape


def test_truncated_step():
    """minimize convex multivariate function with known minimum = 0"""

    def loss(x):
        return jnp.linalg.norm(x)

    x0 = jnp.ones(10)
    traj = [x0]
    loss_traj = [loss(traj[-1])]

    for _ in range(100):
        x = traj[-1]
        f_x = loss(x)
        g_x = grad(loss)(x)

        # if gradient magnitude is small, success has been attained!
        # and we don't want to confuse truncated_step by passing in an all-zeroes search direction
        if np.allclose(g_x, 0):
            break

        # compute step
        x_increment = truncated_step(x, f_x, g_x, step_lower_bound=0.8 * f_x)

        # update trajectory
        x_next = jnp.array(x + x_increment)
        traj.append(x_next)
        loss_traj.append(loss(x_next))

    np.testing.assert_almost_equal(loss_traj[-1], 0.0)
