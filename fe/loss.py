import jax
from jax import core
import jax.numpy as jnp
import numpy as onp

from jax.interpreters import ad

from fe import bar as tmbar
import pymbar
from fe import math_utils


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

    complex_dG = leg_dG(complex_insertion_du_dls, complex_deletion_du_dls, lambda_schedule)
    solvent_dG = leg_dG(solvent_insertion_du_dls, solvent_deletion_du_dls, lambda_schedule)

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
    true_dG):

    complex_dG = EXP(complex_du_dls, lambda_schedule)
    solvent_dG = EXP(solvent_du_dls, lambda_schedule)

    print("complex_dG", complex_dG)
    print("solvent_dG", solvent_dG)

    pred_dG = solvent_dG - complex_dG
    loss = jnp.power(true_dG - pred_dG, 2)

    return loss