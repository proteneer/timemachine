from optimize.step import truncated_step

import numpy as onp

from jax import numpy as jnp
from jax import grad
from jax.config import config
config.update("jax_enable_x64", True)


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
        if onp.allclose(g_x, 0):
            break

        # compute step
        x_increment = truncated_step(x, f_x, g_x, step_lower_bound=0.8 * f_x)

        # update trajectory
        x_next = jnp.array(x + x_increment)
        traj.append(x_next)
        loss_traj.append(loss(x_next))

    onp.testing.assert_almost_equal(loss_traj[-1], 0.0)
