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

class LambdaPotential(CustomOpWrapper):

    def bind(self, params):
        raise ValueError("LambdaPotential cannot bind parameters")

    def get_u_fn(self):
        return self.args[0]

    def set_N(self, N):
        self.args[1] = N

    def get_N(self):
        return self.args[1]

    def get_multiplier(self):
        return self.args[3]

    def get_offset(self):
        return self.args[4]

    def unbound_impl(self, precision):
        return custom_ops.LambdaPotential(
            self.args[0].unbound_impl(precision),
            *self.args[1:]
        )

    def bound_impl(self, precision):
        u_params = self.get_u_fn().params
        return custom_ops.BoundPotential(
            self.unbound_impl(precision),
            u_params
        )

class Shape(CustomOpWrapper):

    def get_N(self):
        return self.args[0]

    def get_a_idxs(self):
        return self.args[1]

    def get_b_idxs(self):
        return self.args[2]


class HarmonicBond(CustomOpWrapper):

    def get_bond_idxs(self):
        return self.args[0]

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

class HarmonicAngle(CustomOpWrapper):

    def get_angle_idxs(self):
        return self.args[0]

class PeriodicTorsion(CustomOpWrapper):

    def get_torsion_idxs(self):
        return self.args[0]

class CentroidRestraint(CustomOpWrapper):

    def get_a_idxs(self):
        return self.args[0]

    def get_b_idxs(self):
        return self.args[1]

    def get_masses(self):
        return self.args[2]

    def set_masses(self, masses):
        self.args[2] = masses


class NonbondedCustomOpWrapper(CustomOpWrapper):

    def __init__(self, *args):

        # exclusion_idxs should be unique
        exclusion_idxs = args[0]
        exclusion_set = set()

        for src, dst in exclusion_idxs:
            src, dst = sorted((src, dst))
            exclusion_set.add((src, dst))

        assert len(exclusion_set) == exclusion_idxs.shape[0]

        super(NonbondedCustomOpWrapper, self).__init__(*args)

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

    def get_beta(self):
        return self.args[4]

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


