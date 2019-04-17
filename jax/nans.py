import time
import numpy as vnp
from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np


import scipy.stats as stats


def dij(ci, cj):
    dx = ci - cj
    d = np.linalg.norm(dx)
    return d



dxdi = jax.jacfwd(dij, argnums=(0,))

a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.0, 3.5])

g = dxdi(a, b)

print(g)