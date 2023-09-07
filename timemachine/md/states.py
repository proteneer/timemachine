# Sampler states

from collections import namedtuple

CoordsBox = namedtuple("CoordsBox", ["coords", "box"])

CoordsVelBox = namedtuple("CoordsVelBox", ["coords", "velocities", "box"])
