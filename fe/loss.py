import jax
from jax import core
import jax.numpy as jnp
import numpy as onp

from jax.interpreters import ad

from fe import bar as tmbar
import pymbar
from fe import math_utils

from timemachine.constants import kB
from simtk import unit


# (ytz): the AD override trick is taken from:
# https://github.com/google/jax/issues/1142
# courtesy of mattjj
def mybar_impl(w):
    A, _ = pymbar.BAR(w[0], w[1])
    return A

def mybar_vjp(g, w):
    return g*tmbar.dG_dw(w)

def mybar(x):
    return mybar_p.bind(x)

mybar_p = core.Primitive('mybar')
mybar_p.def_impl(mybar_impl)
ad.defvjp(mybar_p, mybar_vjp)


def BAR_leg(
    insertion_du_dls,
    deletion_du_dls,
    lambda_schedule):
    insertion_W = math_utils.trapz(insertion_du_dls, lambda_schedule)
    deletion_W = math_utils.trapz(deletion_du_dls, lambda_schedule)

    return mybar(jnp.stack([insertion_W, deletion_W]))
   
def BAR_loss(
    complex_insertion_du_dls, # [C, N]
    complex_deletion_du_dls,  # [C, N]
    solvent_insertion_du_dls, # [C, N]
    solvent_deletion_du_dls,  # [C, N]
    lambda_schedule,
    true_dG):

    complex_dG = BAR_leg(complex_insertion_du_dls, complex_deletion_du_dls, lambda_schedule)
    solvent_dG = BAR_leg(solvent_insertion_du_dls, solvent_deletion_du_dls, lambda_schedule)

    pred_dG = solvent_dG - complex_dG
    loss = jnp.power(true_dG - pred_dG, 2)

    return loss

def EXP_from_du_dls(
    all_du_dls,
    lambda_schedule,
    kT):
    """
    Run exponential averaging on a list of du_dls that may contain None elements.

    The inputs for du_dls should be in units of 1/kT
    """
    proper_du_dls = []

    for d in all_du_dls:
        if d is not None:
            proper_du_dls.append(d)

    proper_du_dls = jnp.array(proper_du_dls)

    work_array = math_utils.trapz(proper_du_dls, lambda_schedule)
    work_array = work_array/kT

    return tmbar.EXP(work_array)*kT
   
def EXP_loss(
    complex_du_dls, # [C, N]
    solvent_du_dls, # [C, N]
    lambda_schedule,
    true_dG,
    temperature=300 * unit.kelvin):

    kT = kB * temperature

    complex_dG = EXP_from_du_dls(complex_du_dls, lambda_schedule, kT)
    solvent_dG = EXP_from_du_dls(solvent_du_dls, lambda_schedule, kT)

    pred_dG = solvent_dG - complex_dG
    loss = jnp.power(true_dG - pred_dG, 2)

    return loss


def _scalar_compute_residual(prediction, label, reliable_interval=(-jnp.inf, +jnp.inf)):
    """prediction - label if label is in the reliable interval"""

    lower, upper = reliable_interval

    if (label >= lower) and (label <= upper):
        residual = prediction - label
    elif label < lower:
        residual = max(0, prediction - lower)
    elif label > upper:
        residual = min(0, prediction - upper)
    else:
        raise (RuntimeError('unsatisfiable reliable_range'))

    return residual


def l1_loss(residual):
    """loss = abs(residual)"""
    return jnp.abs(residual)


def pseudo_huber_loss(residual, threshold=4.184):
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


def flat_bottom_loss(residual, threshold=4.184):
    """loss = max(0, |residual| - threshold)"""
    return jnp.maximum(0, jnp.abs(residual) - threshold)
