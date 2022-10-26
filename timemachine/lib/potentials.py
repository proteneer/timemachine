from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from timemachine.lib import custom_ops

# (ytz): classes in this class wrap custom_ops but have the added benefit
# of being pickleable.


class CustomOpWrapper:
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


class SummedPotential(CustomOpWrapper):
    def __init__(self, potentials: Sequence[CustomOpWrapper], params_init: Sequence[NDArray]):

        if len(potentials) != len(params_init):
            raise ValueError("number of potentials != number of parameter arrays")

        self._potentials = potentials
        self._params_init = params_init
        self.params = None

    def unbound_impl(self, precision):
        sizes = [ps.size for ps in self._params_init]
        impls = [p.unbound_impl(precision) for p in self._potentials]
        return custom_ops.SummedPotential(impls, sizes)


class FanoutSummedPotential(CustomOpWrapper):
    def __init__(self, potentials: Sequence[CustomOpWrapper]):
        self._potentials = potentials

    def unbound_impl(self, precision):
        impls = [p.unbound_impl(precision) for p in self._potentials]
        return custom_ops.FanoutSummedPotential(impls)


class RMSDRestraint(CustomOpWrapper):
    pass


class BondedWrapper(CustomOpWrapper):
    def get_idxs(self):
        return self.args[0]

    def set_idxs(self, new_idxs):
        self.args[0] = new_idxs


class ChiralAtomRestraint(BondedWrapper):
    def get_idxs(self):
        return self.args[0]

    def set_idxs(self, new_idxs):
        self.args[0] = new_idxs


class ChiralBondRestraint(BondedWrapper):
    def get_idxs(self):
        return self.args[0]

    def set_idxs(self, new_idxs):
        self.args[0] = new_idxs

    def get_signs(self):
        return self.args[1]


class HarmonicBond(BondedWrapper):
    pass


class FlatBottomBond(CustomOpWrapper):
    def get_idxs(self):
        return self.args[0]

    def set_idxs(self, new_idxs):
        self.args[0] = new_idxs


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

    def get_kb(self):
        return self.args[2]

    def get_b0(self):
        return self.args[3]


class NonbondedImplWrapper(custom_ops.FanoutSummedPotential):
    """Wraps custom_ops.FanoutSummedPotential, adding methods that
    should be provided by the Nonbonded implementation according to
    the current API spec"""

    # NOTE: This extends custom_ops.FanoutSummedPotential (vs.
    # CustomOpWrapper) because it needs to implement the C++ Potential
    # class interface. The motivation for this is to allow the
    # Nonbonded kernel as implemented using FanoutSummedPotential to
    # maintain the same interface as the older monolithic kernel, in
    # particular providing disable_hilbert_sort() and
    # set_nblist_padding().
    #
    # Longer term, it might be preferable to make a breaking API
    # change so that e.g. disable_hilbert_sort() isn't called directly
    # on the full nonbonded potential (which is typically a sum of
    # parts, not all of which use a neighborlist), but on the
    # underlying NonbondedAllPairs or NonbondedInteractionGroup
    # potentials (a helper function could be added to the Python
    # library to make this more convenient). At that point, the
    # wrapper class can be removed.

    def disable_hilbert_sort(self):
        for impl in self.get_potentials():
            if hasattr(impl, "disable_hilbert_sort"):
                impl.disable_hilbert_sort()

    def set_nblist_padding(self, padding):
        for impl in self.get_potentials():
            if hasattr(impl, "set_nblist_padding"):
                impl.set_nblist_padding(padding)


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

    def unbound_impl(self, precision):
        all_pairs_impl = NonbondedAllPairs(
            self.get_lambda_plane_idxs(),
            self.get_lambda_offset_idxs(),
            self.get_beta(),
            self.get_cutoff(),
        ).unbound_impl(precision)

        exclusions_impl = NonbondedPairListNegated(
            self.get_exclusion_idxs(),
            self.get_scale_factors(),
            self.get_lambda_plane_idxs(),
            self.get_lambda_offset_idxs(),
            self.get_beta(),
            self.get_cutoff(),
        ).unbound_impl(precision)

        return NonbondedImplWrapper([all_pairs_impl, exclusions_impl])

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
        return self.args[5]


class NonbondedAllPairs(CustomOpWrapper):
    def get_lambda_plane_idxs(self):
        return self.args[0]

    def get_lambda_offset_idxs(self):
        return self.args[1]

    def get_beta(self):
        return self.args[2]

    def get_cutoff(self):
        return self.args[3]

    def get_atom_idxs(self):
        return self.args[4]


class NonbondedInteractionGroup(CustomOpWrapper):
    def get_row_atom_idxs(self):
        return self.args[0]

    def get_lambda_plane_idxs(self):
        return self.args[1]

    def get_lambda_offset_idxs(self):
        return self.args[2]

    def get_beta(self):
        return self.args[3]

    def get_cutoff(self):
        return self.args[4]


class NonbondedPairList(CustomOpWrapper):
    def get_idxs(self):
        return self.args[0]

    def get_rescale_mask(self):
        return self.args[1]

    def get_lambda_plane_idxs(self):
        return self.args[2]

    def get_lambda_offset_idxs(self):
        return self.args[3]

    def get_beta(self):
        return self.args[4]

    def get_cutoff(self):
        return self.args[5]


class NonbondedPairListPrecomputed(CustomOpWrapper):
    """
    This implements a pairlist with precomputed parameters. It differs from
    the regular NonbondedPairlist in that it expects params of the form s0*q_ij, s_ij, and s1*e_ij
    where s are the scaling factor and combining rules have already been applied.

    Note that you should not use this class to implement exclusions (that are later cancelled out by AllPairs)
    since the floating point operations are different in python vs C++.
    """

    def get_idxs(self):
        return self.args[0]

    def set_idxs(self, idxs):
        self.args[0] = idxs

    def get_offsets(self):
        return self.args[1]

    def get_beta(self):
        return self.args[2]

    def get_cutoff(self):
        return self.args[3]


class NonbondedPairListNegated(NonbondedPairList):
    def unbound_impl(self, precision):
        cls_name_base = "NonbondedPairList"
        if precision == np.float64:
            cls_name_base += "_f64_negated"
        else:
            cls_name_base += "_f32_negated"

        custom_ctor = getattr(custom_ops, cls_name_base)

        return custom_ctor(*self.args)
