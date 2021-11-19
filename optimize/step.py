import numpy as np
from jax import numpy as jnp
from ff import nonbonded

from typing import Union, Optional

try:
    from scipy.optimize import root_scalar
except ImportError as error:
    import scipy

    print(f"scipy version is {scipy.__version__}, but `scipy.optimize.root_scalar` was added in 1.2")
    raise error


array = Union[np.array, jnp.array]


def _taylor_first_order(x: array, f_x: float, grad: array) -> callable:
    """

    Notes:
        TODO: is it preferable to use jax linearize? https://jax.readthedocs.io/en/latest/jax.html#jax.linearize
    """

    def f_prime(y: array) -> float:
        return f_x + np.dot(grad, y - x)

    return f_prime


def truncated_step(
    x: array,
    f_x: float,
    grad: array,
    step_size: float = 0.1,
    search_direction: Optional[array] = None,
    step_lower_bound: float = 0.0,
):
    """Motivated by https://arxiv.org/abs/1903.08619 , use knowledge of a lower-bound on f_x
    to prevent from taking a step too large

    TODO: consider further damping?

    TODO: rather than truncating at absolute global bound on loss,
        consider truncating at relative bound, like, don't take a step that
        you predict would decrease the loss by more than
            X % ?
            X absolute increment?

            some combination of these?

    TODO: generalize to use local surrogates other than first-order Taylor expansions
        by refactoring to accept a callable `f_prime` directly,
        rather than constructing a default `f_prime` from `x`, `f_x`, `grad` arguments

    Notes
    -----
    * search_direction not assumed normalized. for example, it could be the raw gradient

    * `step_size` is used to generate an initial proposal `x_proposed`. If `f_prime(x_proposed) < step_lower_bound`,
        then the step will be truncated.

    * The default `step_lower_bound=0` corresponds to a suggestion in the cited study, incorporating the knowledge that
        the loss is bounded below by 0. In the script, we pass in a non-default argument to the `step_lower_bound` to
        make the behavior of the method more conservative, and this is probably something we'll fiddle with a bit.

    * The default value `step_size=0.1` isn't very precisely chosen. The behavior of the method will be insensitive to
        picking `step_size` anywhere between like 1e-3 and +inf for our problems, since this will trigger the
        step-truncating logic on most every step.
        If the `step_size` is chosen sufficiently small that it rarely produces proposals that violate `step_lower_bound`,
        then that will start to have an effect on the behavior of the optimizer.

    """

    # default search direction: SGD
    if search_direction is None:
        search_direction = -grad
    assert np.linalg.norm(search_direction) > 0  # if this vector is all zeros, doesn't make sense to proceed

    # default local surrogate model: linear
    f_prime = _taylor_first_order(x, f_x, grad)

    # default step: step_size * search_direction
    x_next = x + step_size * search_direction

    # if this is too optimistic, according to local surrogate f_prime
    if f_prime(x_next) < step_lower_bound:  # TODO: replace f_prime bound with something more configurable
        x_proposed = x_next

        line_search_fxn = lambda alpha: f_prime(x + alpha * search_direction) - step_lower_bound

        result = root_scalar(line_search_fxn, x0=0, x1=step_size)
        alpha = result.root

        x_next = x + alpha * search_direction

        message = f"""
        f_prime(x_proposed) = {f_prime(x_proposed):.5f}
        using default step size {step_size:.5f}
        is lower than step_lower_bound = {step_lower_bound:.5f}

        truncating step size to {alpha:.5f},
        so that the predicted f_prime(x_next) = {f_prime(x_next):.5f}"""
        print(message)

    x_increment = np.array(x_next - x)

    return x_increment


# TODO: define more flexible update rules here, rather than update parameters
step_sizes = {
    nonbonded.AM1CCCHandler: 1e-3,
    nonbonded.LennardJonesHandler: 1e-3,
    # ...
}

gradient_clip_thresholds = {
    nonbonded.AM1CCCHandler: 0.001,
    nonbonded.LennardJonesHandler: np.array([0.001, 0]),  # TODO: allow to update epsilon also?
    # ...
}


def _clipped_update(gradient, step_size, clip_threshold):
    """Compute an update based on current gradient
        x[k+1] = x[k] + update

    The gradient descent update would be
        update = - step_size * grad(x[k]),

    and to avoid instability, we clip the absolute values of all components of the update
        update = - clip(step_size * grad(x[k]))

    TODO: menu of other, fancier update functions
    """
    return -np.clip(step_size * gradient, -clip_threshold, clip_threshold)
