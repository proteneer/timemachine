import jax.numpy as jnp
import pymbar
from jax import core
from jax.interpreters import ad

from timemachine.constants import KCAL_TO_KJ
from timemachine.fe import bar as tmbar


# (ytz): the AD override trick is taken from:
# https://github.com/google/jax/issues/1142
# courtesy of mattjj
def mybar_impl(w):
    A, _ = pymbar.BAR(w[0], w[1])
    return A


def mybar_jvp(g, w):
    return g * tmbar.dG_dw(w)


def mybar(x):
    return mybar_p.bind(x)


mybar_p = core.Primitive("mybar")
mybar_p.def_impl(mybar_impl)
ad.defjvp(mybar_p, mybar_jvp)


def BAR_leg(w_insert, w_delete):
    """

    Parameters
    ----------
    w_insert, w_delete: arrays
        works in reduced units

    Returns
    -------
    dG : float
    """
    return mybar(jnp.stack([w_insert, w_delete]))


def BAR_loss(
    complex_w_insert,
    complex_w_delete,
    solvent_w_insert,
    solvent_w_delete,
    true_delta_f,
):
    """

    Parameters
    ----------
    complex_w_insert, complex_w_delete, solvent_w_insert, solvent_w_delete
        work arrays (in reduced units)
    true_delta_f : float
        in reduced units

    Returns
    -------
    squared_loss : float
    """

    complex_delta_f = BAR_leg(complex_w_insert, complex_w_delete)
    solvent_delta_f = BAR_leg(solvent_w_insert, solvent_w_delete)

    pred_delta_f = solvent_delta_f - complex_delta_f
    loss = (true_delta_f - pred_delta_f) ** 2

    return loss


def EXP_loss(complex_w_insert, solvent_w_insert, true_delta_f):
    complex_delta_f = tmbar.EXP(complex_w_insert)
    solvent_delta_f = tmbar.EXP(solvent_w_insert)

    pred_delta_f = solvent_delta_f - complex_delta_f
    loss = (true_delta_f - pred_delta_f) ** 2

    return loss


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
