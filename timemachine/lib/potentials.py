import numpy as np
from timemachine.lib import custom_ops

# (ytz): classes in this class wrap custom_ops but have the added benefit
# of being pickleable.

class CustomOpWrapper():

    def __init__(self, *args, precision):
        self.args = args
        self.precision = precision

    def impl(self):
        cls_name_base = type(self).__name__
        if self.precision == np.float64:
            cls_name_base += "_f64"
        else:
            cls_name_base += "_f32"

        custom_ctor = getattr(custom_ops, cls_name_base)

        return custom_ctor(*self.args)

    def bind_impl(self, params):
        """ 

        Bind the potential to the given set of parameters. 

        """
        p = self.impl()
        # p is reference collected and tossed out
        return custom_ops.BoundPotential(p, params)

class HarmonicBond(CustomOpWrapper):
    pass

class HarmonicAngle(CustomOpWrapper):
    pass

class PeriodicTorsion(CustomOpWrapper):
    pass


class NonbondedCustomOpWrapper(CustomOpWrapper):

    def __init__(self, *args, precision):

        # exclusion_idxs should be unique
        exclusion_idxs = args[0]
        exclusion_set = set()

        for src, dst in exclusion_idxs:
            src, dst = sorted((src, dst))
            exclusion_set.add((src, dst))

        assert len(exclusion_set) == exclusion_idxs.shape[0]

        super(NonbondedCustomOpWrapper, self).__init__(*args, precision=precision)

class LennardJones(NonbondedCustomOpWrapper):
    pass

class Electrostatics(NonbondedCustomOpWrapper):
    pass
