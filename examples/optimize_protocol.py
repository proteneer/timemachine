from jax.config import config;

config.update("jax_enable_x64", True)
from jax import grad, hessian, jit, vmap, numpy as np
import jax

from collections import namedtuple
from functools import partial

cutoff = 5.0


# force field terms

def lennard_jones(r: float, sigma: float, epsilon: float) -> float:
    """https://en.wikipedia.org/wiki/Lennard-Jones_potential"""
    return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)


def coulomb(r: float, charge_product: float) -> float:
    """https://en.wikipedia.org/wiki/Electric_potential"""
    return charge_product / r


Geometry = namedtuple('Geometry', ['r'])
FFParams = namedtuple('FFParams', ['sigma', 'epsilon', 'charge_product'])
ControlDials = namedtuple('ControlDials', ['lj_offset', 'coulomb_offset'])


def u_controllable(x: Geometry, ff: FFParams, dials: ControlDials) -> float:
    """Separately controllable distance offsets for LJ and Coulomb"""
    u_lj = lennard_jones(x.r + dials.lj_offset, ff.sigma, ff.epsilon)
    u_coulomb = coulomb(x.r + dials.coulomb_offset, ff.charge_product)

    return u_lj + u_coulomb
