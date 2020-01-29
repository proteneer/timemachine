import numpy as np
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import pymbar
import jax

def EXP(w):
    """
    Estimate free energy difference using exponential averaging

    Parameters
    ----------
    w : np.ndarray, float, (2, N)
        forward and reverse work for N frames

    Returns
    ------
    deltaF : scalar, float
        free energy difference

    """
    T = float(jnp.size(w))
    deltaF = - (logsumexp(-w) - jnp.log(T))
    return deltaF

def BARzero(w, deltaF):
    """
    A function that when zeroed is equivalent to the solution of BAR

    Parameters
    ---------
    w : np.ndarray, float, (2, N)
        forward and reverse work for N frames

    deltaF : scalar, float
        free energy difference

    Returns
    ------
    fzero : scalar, float
        a variable that is zeroed when deltaF satisfies BAR

    """
    w_F = w[0]
    w_R = w[1]
    T_F = len(w_F)
    T_R = len(w_R)
    M = jnp.log(T_F / T_R)

    exp_arg_F = (M + w_F - deltaF)
    max_arg_F = jnp.where(exp_arg_F < 0, 0.0, exp_arg_F)

    log_f_F = - max_arg_F - jnp.log(jnp.exp(-max_arg_F) + jnp.exp(exp_arg_F - max_arg_F))
    log_numer = logsumexp(log_f_F)
    exp_arg_R = -(M - w_R - deltaF)
    max_arg_R = jnp.where(exp_arg_R < 0, 0.0, exp_arg_R)

    log_f_R = - max_arg_R - jnp.log(jnp.exp(-max_arg_R) + jnp.exp(exp_arg_R - max_arg_R))

    log_denom = logsumexp(log_f_R)
    fzero = log_numer - log_denom
    return fzero

def dG_dw(w):
    """
    A function that calculates gradient of free energy difference with respect to work

    Parameters
    ---------
    w : np.ndarray, float, (2, N)

    Returns
    ------
    dG_dw : np.ndarray, float, (2, N)
        The gradient of free energy difference with respect to work

    """
    dG, _ = pymbar.BAR(w[0], w[1])
    dBAR_dw = jax.grad(BARzero, argnums=(0,))
    dBAR_dA = jax.grad(BARzero, argnums=(1,))
    dG_dw = -dBAR_dw(w,dG)[0]/dBAR_dA(w,dG)[0]
    return dG_dw
