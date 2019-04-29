import time
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np


def fog(x):
    # h(x) = (fog)(x) = (2x)^2
    g = 2*x # accumulated tangent: dg/dx = 2 
    fg = g**2 # accumulated tangent: df/dg*dg/dx = 2*2x*2 = 8x
    # in principle with fwd mode autodiff I should be able to inspect g.tangent fg.tangent
    # to actually inspect the both parts of the dual.

    # what do I do with these tangents to inspect their valuers?
    print("??", g.tangent)
    print("??", fg.tangent)

    return fg

dfog_dx = jax.jacfwd(fog, argnums=(0,))
print("--", jax.make_jaxpr(dfog_dx)(3.0))
print(dfog_dx(np.array([3.0])))