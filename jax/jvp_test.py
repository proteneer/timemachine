import time
import numpy as vnp
from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np

from jax import custom_transforms
from jax.interpreters.ad import defjvp

def print_tangent_jvp(t, x):
  print(t)
  return t

print_tangent = custom_transforms(lambda x: x)
defjvp(print_tangent.primitive, print_tangent_jvp)

import scipy.stats as stats

def fog(x):
    return np.sum(np.cos(np.power(x, 2.0)))


if __name__ == "__main__":


    theta = np.array([25000.0, 0.129], dtype=np.float64)
    grads = np.array([1.0, 0.0], dtype=np.float64)

    primals, tangents = jax.jvp(fog, (theta,), (grads,))

    print("primals", primals)
    print("tangents", tangents)

    grads = np.array([0.0, 1.0], dtype=np.float64)

    primals, tangents = jax.jvp(fog, (theta,), (grads,))

    print("primals", primals)
    print("tangents", tangents)
