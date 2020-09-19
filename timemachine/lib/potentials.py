import numpy as np
from timemachine.lib import custom_ops

# (ytz): classes in this class wrap custom_ops but have the added benefit
# of being pickleable.

# see test_binding.py for example usage


# class BoundPotentialWrapper():
    
#     def __init__(self, wrapped_custom_op, params):
#         self.wrapped_custom_op = wrapped_custom_op
#         self.params = params

#     def get_op(self):
#         return self.wrapped_custom_op

#     def impl(self):
#         return custom_ops.BoundPotential(
#             self.wrapped_custom_op.impl(),
#             self.params
#         )




class CustomOpWrapper():

    def __init__(self, *args, precision):
        self.args = args
        self.precision = precision
        self.params = None

    def bind(self, params):
        self.params = params
        # return self is to allow chaining
        return self

    def unbound_impl(self):
        cls_name_base = type(self).__name__
        if self.precision == np.float64:
            cls_name_base += "_f64"
        else:
            cls_name_base += "_f32"

        custom_ctor = getattr(custom_ops, cls_name_base)

        return custom_ctor(*self.args)

    def bound_impl(self):
        if self.params is None:
            raise ValueError("This op has not been bound to parameters.")

        return custom_ops.BoundPotential(self.unbound_impl(), self.params)


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

    def get_exclusion_idxs(self):
        return self.args[0]

    def get_scale_factors(self):
        return self.args[1]

    def get_lambda_plane_idxs(self):
        return self.args[2]

    def get_lambda_offset_idxs(self):
        return self.args[3]

    def get_cutoff(self):
        return self.args[-1]

class Nonbonded(NonbondedCustomOpWrapper):
    pass

class LennardJones(NonbondedCustomOpWrapper):
    pass

class Electrostatics(NonbondedCustomOpWrapper):

    def get_beta(self):
        return self.args[4]
    pass


