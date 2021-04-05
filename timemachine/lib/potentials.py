import numpy as np
from timemachine.lib import custom_ops

# (ytz): classes in this class wrap custom_ops but have the added benefit
# of being pickleable.

BoundPotential = custom_ops.BoundPotential

class CustomOpWrapper():

    def __init__(self, *args):
        # needed in case we need to modify the args.
        self.args = list(args)
        self.params = None

    def bind(self, params):
        self.params = params
        # return self is to allow chaining
        return self

    def unbound_impl(self, precision):
        cls_name_base = type(self).__name__
        if precision == np.float64:
            cls_name_base += "_f64"
        else:
            cls_name_base += "_f32"

        custom_ctor = getattr(custom_ops, cls_name_base)

        return custom_ctor(*self.args)

    def bound_impl(self, precision):
        if self.params is None:
            raise ValueError("This op has not been bound to parameters.")

        return custom_ops.BoundPotential(self.unbound_impl(precision), self.params)


# should not be used for Nonbonded Potentials
# class InterpolatedPotential(CustomOpWrapper):

#     def bind(self, params):
#         assert self.get_u_fn().params is None
#         self.params = params
#         return self

#     def get_u_fn(self):
#         return self.args[0]

#     def unbound_impl(self, precision):
#         return custom_ops.InterpolatedPotential(
#             self.get_u_fn().unbound_impl(precision),
#             self.args[1],
#             self.args[2]
#         )

#     def bound_impl(self, precision):
#         return custom_ops.BoundPotential(
#             self.unbound_impl(precision),
#             self.params
#         )

# class LambdaPotential(CustomOpWrapper):

#     def bind(self, params):
#         assert self.get_u_fn().params is None
#         self.params = params
#         return self

#     # pass through so we can use the underlying methods
#     def __getattr__(self, attr):
#         return getattr(self.args[0], attr)

#     def get_u_fn(self):
#         return self.args[0]

#     def set_N(self, N):
#         self.args[1] = N

#     def get_N(self):
#         return self.args[1]

#     def get_multiplier(self):
#         return self.args[3]

#     def get_offset(self):
#         return self.args[4]

#     def unbound_impl(self, precision):
#         return custom_ops.LambdaPotential(
#             self.args[0].unbound_impl(precision),
#             *self.args[1:]
#         )

#     def bound_impl(self, precision):
#         u_params = self.get_u_fn().params
#         return custom_ops.BoundPotential(
#             self.unbound_impl(precision),
#             self.params
#         )

# class Shape(CustomOpWrapper):

#     def get_N(self):
#         return self.args[0]

#     def get_a_idxs(self):
#         return self.args[1]

#     def get_b_idxs(self):
#         return self.args[2]

class RMSDRestraint(CustomOpWrapper):
    pass


class BondedWrapper(CustomOpWrapper):

    def get_idxs(self):
        return self.args[0]

    def set_idxs(self, new_idxs):
        self.args[0] = new_idxs

    def get_lambda_mult(self):
        if len(self.args) > 1:
            return self.args[1]
        else:
            return None

    def get_lambda_offset(self):
        if len(self.args) > 1:
            return self.args[2]
        else:
            return None


class HarmonicBond(BondedWrapper):
    pass

# this is an alias to make type checking easier
class CoreRestraint(HarmonicBond):

    def unbound_impl(self, precision):
        cls_name_base = "HarmonicBond"
        if precision == np.float64:
            cls_name_base += "_f64"
        else:
            cls_name_base += "_f32"

        custom_ctor = getattr(custom_ops, cls_name_base)

        return custom_ctor(*self.args)

class HarmonicAngle(BondedWrapper):
    pass


class PeriodicTorsion(BondedWrapper):
    pass

class InertialRestraint(CustomOpWrapper):

    def get_a_idxs(self):
        return self.args[0]

    def get_b_idxs(self):
        return self.args[1]

    def get_masses(self):
        return self.args[2]

    def set_masses(self, masses):
        self.args[2] = masses


class CentroidRestraint(CustomOpWrapper):

    def get_a_idxs(self):
        return self.args[0]

    def get_b_idxs(self):
        return self.args[1]


class Nonbonded(CustomOpWrapper):

    def __init__(self, *args):

        # exclusion_idxs should be unique
        exclusion_idxs = args[0]
        exclusion_set = set()

        for src, dst in exclusion_idxs:
            src, dst = sorted((src, dst))
            exclusion_set.add((src, dst))

        assert len(exclusion_set) == exclusion_idxs.shape[0]

        super(Nonbonded, self).__init__(*args)

    def set_exclusion_idxs(self, x):
        self.args[0] = x

    def get_exclusion_idxs(self):
        return self.args[0]

    def set_scale_factors(self, x):
        self.args[1] = x

    def get_scale_factors(self):
        return self.args[1]

    def get_lambda_plane_idxs(self):
        return self.args[2]

    def get_lambda_offset_idxs(self):
        return self.args[3]

    def set_lambda_plane_idxs(self, val):
        self.args[2] = val

    def set_lambda_offset_idxs(self, val):
        self.args[3] = val

    def get_beta(self):
        return self.args[4]

    def get_cutoff(self):
        return self.args[-1]

class NonbondedInterpolated(Nonbonded):

    def unbound_impl(self, precision):
        cls_name_base = "Nonbonded"
        if precision == np.float64:
            cls_name_base += "_f64_interpolated"
        else:
            cls_name_base += "_f32_interpolated"

        custom_ctor = getattr(custom_ops, cls_name_base)

        return custom_ctor(*self.args)