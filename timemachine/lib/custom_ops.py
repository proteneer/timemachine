# This file contains stubs for some classes defined in the C++
# extension module.
#
# The purpose is to allow importing modules with
# unavoidable top-level references to objects defined in the extension
# (e.g. modules containing subclasses of classes defined in the C++
# code).
#
# If the extension module .so file is present, the definitions
# in it will take precedence over the stubs defined here.


class Context:
    pass


class Potential:
    pass


class BoundPotential:
    pass


class FanoutSummedPotential:
    pass


class TIBDExchangeMove_f32:
    pass


class TIBDExchangeMove_f64:
    pass
