import jax
from jax import core
import jax.numpy as jnp
import numpy as onp

from jax.interpreters import ad

from fe import bar as tmbar
import pymbar
import math_utils


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


def leg_dG(
    insertion_du_dls,
    deletion_du_dls,
    lambda_schedule):
    insertion_W = math_utils.trapz(insertion_du_dls, lambda_schedule)
    deletion_W = math_utils.trapz(deletion_du_dls, lambda_schedule)

    return mybar(jnp.stack([insertion_W, deletion_W]))
   
def loss_dG(
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

    print(solvent_dG, complex_dG, pred_dG, true_dG)

    return loss

# if __name__ == "__main__":

#     C = 128
#     T = 10000
    
#     lambda_schedule = onp.linspace(0.0001, 0.9999, T)
#     cid = onp.stack(onp.random.rand(C, T))
#     cdd = onp.stack(onp.random.rand(C, T))
#     sid = onp.stack(onp.random.rand(C, T))
#     sdd = onp.stack(onp.random.rand(C, T))

#     res = math_utils.trapz(cid, lambda_schedule)

#     loss_grad_fn = jax.grad(loss_dG, argnums=(0,1,2,3))
#     cid_grad, cdd_grad, sid_grad, sdd_grad = loss_grad_fn(cid, cdd, sid, sdd, lambda_schedule, 2.5)
