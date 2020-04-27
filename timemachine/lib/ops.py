import numpy as np
from timemachine.lib import custom_ops

def Nonbonded(*args, precision):
    if precision == np.float64:
        return custom_ops.Nonbonded_f64_3d(*args)
    elif precision == np.float32:
        return custom_ops.Nonbonded_f32_3d(*args)
    else:
        raise Exception("Unknown precision")

def GBSA(*args, precision):
    if precision == np.float64:
        return custom_ops.GBSA_f64_3d(*args)
    elif precision == np.float32:
        return custom_ops.GBSA_f32_3d(*args)
    else:
        raise Exception("Unknown precision")

def HarmonicBond(*args, precision):
    if precision == np.float64:
        return custom_ops.HarmonicBond_f64_3d(*args)
    elif precision == np.float32:
        return custom_ops.HarmonicBond_f32_3d(*args)
    else:
        raise Exception("Unknown precision")

def HarmonicAngle(*args, precision):
    if precision == np.float64:
        return custom_ops.HarmonicAngle_f64_3d(*args)
    elif precision == np.float32:
        return custom_ops.HarmonicAngle_f32_3d(*args)
    else:
        raise Exception("Unknown precision")

def PeriodicTorsion(*args, precision):
    if precision == np.float64:
        return custom_ops.PeriodicTorsion_f64_3d(*args)
    elif precision == np.float32:
        return custom_ops.PeriodicTorsion_f32_3d(*args)
    else:
        raise Exception("Unknown precision")