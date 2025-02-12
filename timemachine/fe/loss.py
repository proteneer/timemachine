import jax.numpy as jnp

from timemachine.constants import KCAL_TO_KJ


def truncated_residuals(predictions, labels, reliable_interval=(-jnp.inf, +jnp.inf)):
    """Adapt "predictions - labels" for cases where labels are only reliable
    within some interval (e.g. when fitting to a "bottomed-out" assay).

    Example
    -------
    >>> labels = jnp.array([0.5, 0.5, 0.5, -6, -6, -6])
    >>> predictions = jnp.array([-10, 0, +10, -10, 0, +10])
    >>> reliable_interval = (-5, +1)
    >>> print(truncated_residuals(predictions, labels, reliable_interval))
    [-10.5  -0.5   9.5   0.    5.   15. ]
    """

    lower, upper = reliable_interval

    residuals = predictions - labels
    residuals = jnp.where(labels < lower, jnp.maximum(0, predictions - lower), residuals)
    residuals = jnp.where(labels > upper, jnp.minimum(0, predictions - upper), residuals)
    return residuals


def l1_loss(residual):
    """loss = abs(residual)"""
    return jnp.abs(residual)


def pseudo_huber_loss(residual, threshold=KCAL_TO_KJ):
    """loss = threshold * (sqrt(1 + (residual/threshold)^2) - 1)

    Reference : https://en.wikipedia.org/wiki/Huber_loss#Pseudo-Huber_loss_function

    Notable properties:
        * As with Huber loss, behaves ~ like L1 above threshold, and ~ like L2 below threshold
            * Note: this means that when |residual| < threshold, the gradient magnitude is lower than with L1 loss
        * Continuous derivatives

    Default value of threshold: 1 kcal/mol, in units of kJ/mol
    """

    # note: the expression quoted on wikipedia will result in slope = threshold -- rather than slope = 1 as desired --
    #   when residual >> threshold
    # return threshold**2 * (np.sqrt(1 + (residual/threshold)**2) - 1)

    # expression used: replace `threshold**2` with `threshold`
    return threshold * (jnp.sqrt(1 + (residual / threshold) ** 2) - 1)


def flat_bottom_loss(residual, threshold=KCAL_TO_KJ):
    """loss = max(0, |residual| - threshold)"""
    return jnp.maximum(0, jnp.abs(residual) - threshold)
