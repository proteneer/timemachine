# Sampler states

from collections import namedtuple
from timemachine.lib import custom_ops

CoordsVelBox = namedtuple("CoordsVelBox", ["coords", "velocities", "box"])


def set_coords_vel_box(ctxt: custom_ops.Context, x: CoordsVelBox):
    ctxt.set_x_t(x.coords)
    ctxt.set_v_t(x.velocities)
    ctxt.set_box(x.box)


def get_coords_vel_box(ctxt: custom_ops.Context) -> CoordsVelBox:
    x_t = ctxt.get_x_t()
    v_t = ctxt.get_v_t()
    box = ctxt.get_box()
    return CoordsVelBox(x_t, v_t, box)
