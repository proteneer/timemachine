import numpy as np
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import pymbar
import jax

def EXP(w):
    T = float(jnp.size(w))
    deltaF = - (logsumexp(-w) - jnp.log(T))
    return deltaF

def BARzero(w, deltaF):
    w_F = w[0]
    w_R = w[1]
    T_F = len(w_F)
    T_R = len(w_R)
    M = jnp.log(T_F / T_R)

    exp_arg_F = (M + w_F - deltaF)
    max_arg_F = jnp.where(exp_arg_F < 0, 0.0, exp_arg_F)

    # try:
    log_f_F = - max_arg_F - jnp.log(jnp.exp(-max_arg_F) + jnp.exp(exp_arg_F - max_arg_F))
    # except:
    #    return np.nan
    log_numer = logsumexp(log_f_F)
    exp_arg_R = -(M - w_R - deltaF)
    max_arg_R = jnp.where(exp_arg_R < 0, 0.0, exp_arg_R)

    #try:
    log_f_R = - max_arg_R - jnp.log(jnp.exp(-max_arg_R) + jnp.exp(exp_arg_R - max_arg_R))
    # except:
    # print("The input data results in overflow in BAR")
    # return np.nan

    log_denom = logsumexp(log_f_R)
    fzero = log_numer - log_denom
    return fzero

def dG_dw(w):
    dG, _ = pymbar.BAR(w[0], w[1])
    dBAR_dw = jax.grad(BARzero, argnums=(0,))
    dBAR_dA = jax.grad(BARzero, argnums=(1,))
    dG_dw = -dBAR_dw(w,dG)[0]/dBAR_dA(w,dG)[0]
    return dG_dw
