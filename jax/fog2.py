import jax
from jax import numpy as np
from jax import custom_transforms
from jax.interpreters.ad import defjvp

def print_tangent_jvp(t, x):
  print(t)
  return t
print_tangent = custom_transforms(lambda x: x)
defjvp(print_tangent.primitive, print_tangent_jvp)

def fog(x):
  g = 2*x
  fg = g**2
  hfg = np.sin(fg)

  print_tangent(g)
  print_tangent(fg)
  print_tangent(hfg)

  return fg

out, out_tangent = jax.jvp(fog, (3.,), (1.,))
