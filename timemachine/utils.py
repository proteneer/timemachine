from pathlib import Path

import timemachine
from timemachine.fe import functional
from timemachine.ff import Forcefield
from timemachine.ff.handlers.deserialize import deserialize_handlers


def get_ff_am1ccc():
    tm_path = Path(timemachine.__path__[0]).parent
    path_to_ff = tm_path / "timemachine/ff/params/smirnoff_1_1_0_ccc.py"
    with open(path_to_ff, "r") as f:
        ff_handlers = deserialize_handlers(f.read())
    ff = Forcefield(ff_handlers)
    return ff


def bind_potentials(ubps, params):
    """modifies ubps in-place"""
    for u, p in zip(ubps, params):
        u.bind(p)


def construct_potential(ubps, params):
    U_fn = functional.construct_differentiable_interface_fast(ubps, params)

    def potential(xvb, lam):
        return U_fn(xvb.coords, params, xvb.box, lam)

    return potential
