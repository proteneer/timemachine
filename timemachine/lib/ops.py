import numpy as np
from timemachine.lib import custom_ops

def Nonbonded(*args, precision):
    dim = args[-1]
    if precision == np.float64:
        if dim == 3:
            return custom_ops.Nonbonded_f64_3d(*args[:-1])
        elif dim == 4:
            return custom_ops.Nonbonded_f64_4d(*args[:-1])
        else:
            raise Exception("Bad Dim")
    elif precision == np.float32:
        if dim == 3:
            return custom_ops.Nonbonded_f32_3d(*args[:-1])
        elif dim == 4:
            return custom_ops.Nonbonded_f32_4d(*args[:-1])
        else:
            raise Exception("Bad Dim")
    else:
        raise Exception("Unknown precision")

def GBSA(*args, precision):
    dim = args[-1]
    if precision == np.float64:
        if dim == 3:
            return custom_ops.GBSA_f64_3d(*args[:-1])
        elif dim == 4:
            return custom_ops.GBSA_f64_4d(*args[:-1])
        else:
            raise Exception("Bad Dim")
    elif precision == np.float32:
        if dim == 3:
            print("32 bit 3D")
            return custom_ops.GBSA_f32_3d(*args[:-1])
        elif dim == 4:
            print("32 bit 4D")
            return custom_ops.GBSA_f32_4d(*args[:-1])
        else:
            raise Exception("Bad Dim")
    else:
        raise Exception("Unknown precision")



def HarmonicBond(*args, precision):
    dim = args[-1]
    if precision == np.float64:
        if dim == 3:
            return custom_ops.HarmonicBond_f64_3d(*args[:-1])
        elif dim == 4:
            return custom_ops.HarmonicBond_f64_4d(*args[:-1])
        else:
            raise Exception("Bad Dim")
    elif precision == np.float32:
        if dim == 3:
            return custom_ops.HarmonicBond_f32_3d(*args[:-1])
        elif dim == 4:
            return custom_ops.HarmonicBond_f32_4d(*args[:-1])
        else:
            raise Exception("Bad Dim")
    else:
        raise Exception("Unknown precision")

def HarmonicAngle(*args, precision):
    dim = args[-1]
    if precision == np.float64:
        if dim == 3:
            return custom_ops.HarmonicAngle_f64_3d(*args[:-1])
        elif dim == 4:
            return custom_ops.HarmonicAngle_f64_4d(*args[:-1])
        else:
            raise Exception("Bad Dim")
    elif precision == np.float32:
        if dim == 3:
            return custom_ops.HarmonicAngle_f32_3d(*args[:-1])
        elif dim == 4:
            return custom_ops.HarmonicAngle_f32_4d(*args[:-1])
        else:
            raise Exception("Bad Dim")
    else:
        raise Exception("Unknown precision")

def PeriodicTorsion(*args, precision):
    dim = args[-1]
    if precision == np.float64:
        if dim == 3:
            return custom_ops.PeriodicTorsion_f64_3d(*args[:-1])
        elif dim == 4:
            return custom_ops.PeriodicTorsion_f64_4d(*args[:-1])
        else:
            raise Exception("Bad Dim")
    elif precision == np.float32:
        if dim == 3:
            return custom_ops.PeriodicTorsion_f32_3d(*args[:-1])
        elif dim == 4:
            return custom_ops.PeriodicTorsion_f32_4d(*args[:-1])
        else:
            raise Exception("Bad Dim")
    else:
        raise Exception("Unknown precision")