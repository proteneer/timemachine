import numpy as np
from timemachine.lib import custom_ops

def Nonbonded(*args, precision):

    # exclusion_idxs should be unique
    exclusion_idxs = args[2]
    exclusion_set = set()

    for src, dst in exclusion_idxs:
        src, dst = sorted((src, dst))
        exclusion_set.add((src, dst))

    assert len(exclusion_set) == exclusion_idxs.shape[0]


    if precision == np.float64:
        return custom_ops.Nonbonded_f64(*args)
    elif precision == np.float32:
        return custom_ops.Nonbonded_f32(*args)
    else:
        raise Exception("Unknown precision")

def GBSA(*args, precision):
    if precision == np.float64:
        return custom_ops.GBSA_f64(*args)
    elif precision == np.float32:
        return custom_ops.GBSA_f32(*args)
    else:
        raise Exception("Unknown precision")

def HarmonicBond(*args, precision):
    if precision == np.float64:
        return custom_ops.HarmonicBond_f64(*args)
    elif precision == np.float32:
        return custom_ops.HarmonicBond_f32(*args)
    else:
        raise Exception("Unknown precision")

def FlatBottom(*args, precision):
    if precision == np.float64:
        return custom_ops.FlatBottom_f64(*args)
    elif precision == np.float32:
        return custom_ops.FlatBottom_f32(*args)
    else:
        raise Exception("Unknown precision")

def HarmonicAngle(*args, precision):
    if precision == np.float64:
        return custom_ops.HarmonicAngle_f64(*args)
    elif precision == np.float32:
        return custom_ops.HarmonicAngle_f32(*args)
    else:
        raise Exception("Unknown precision")

def PeriodicTorsion(*args, precision):
    if precision == np.float64:
        return custom_ops.PeriodicTorsion_f64(*args)
    elif precision == np.float32:
        return custom_ops.PeriodicTorsion_f32(*args)
    else:
        raise Exception("Unknown precision")

def AlchemicalGradient(*args):
    return custom_ops.AlchemicalGradient(*args)