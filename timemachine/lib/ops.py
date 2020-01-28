import numpy as np
from timemachine.lib import custom_ops

precision = np.float64

def Nonbonded(*args):
    dim = args[-1]
    if dim == 3:
        return custom_ops.Nonbonded_f64_3d(*args[:-1])
    elif dim == 4:
        return custom_ops.Nonbonded_f64_4d(*args[:-1])
    else:
        raise Exception("Bad Dim")


def HarmonicBond(*args):
    dim = args[-1]
    if dim == 3:
        return custom_ops.HarmonicBond_f64_3d(*args[:-1])
    elif dim == 4:
        return custom_ops.HarmonicBond_f64_4d(*args[:-1])
    else:
        raise Exception("Bad Dim")

def HarmonicAngle(*args):
    dim = args[-1]
    if dim == 3:
        return custom_ops.HarmonicAngle_f64_3d(*args[:-1])
    elif dim == 4:
        return custom_ops.HarmonicAngle_f64_4d(*args[:-1])
    else:
        raise Exception("Bad Dim")

def PeriodicTorsion(*args):
    dim = args[-1]
    if dim == 3:
        return custom_ops.PeriodicTorsion_f64_3d(*args[:-1])
    elif dim == 4:
        return custom_ops.PeriodicTorsion_f64_4d(*args[:-1])
    else:
        raise Exception("Bad Dim")