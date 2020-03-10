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

def EXP_leg(
    du_dls,
    lambda_schedule):
    insertion_W = math_utils.trapz(du_dls, lambda_schedule)

    return tmbar.EXP(insertion_W)
   
def EXP_loss(
    complex_du_dls, # [C, N]
    solvent_du_dls, # [C, N]
    lambda_schedule,
    true_dG):

    complex_dG = EXP_leg(complex_du_dls, lambda_schedule)
    solvent_dG = EXP_leg(solvent_du_dls, lambda_schedule)

    print("complex_dG", complex_dG)
    print("solvent_dG", solvent_dG)

    pred_dG = solvent_dG - complex_dG
    loss = jnp.power(true_dG - pred_dG, 2)

    return loss