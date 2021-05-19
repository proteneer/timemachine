# Sampler states

from collections import namedtuple

CoordsAndBox = namedtuple('CoordsAndBox', ['coords', 'box'])
CoordsVelBox = namedtuple('CoordsVelBox', ['coords', 'velocities', 'box'])