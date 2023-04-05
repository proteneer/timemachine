import ast

from timemachine import constants
from timemachine.ff.handlers import bonded, nonbonded, serialization_format
from timemachine.ff.handlers.suffix import _SUFFIX


def deserialize_handlers(obj):
    """
    Parameters
    ----------
    obj: bytes-like
        the binary we wish to deserialize.

    Returns
    -------
    a handler from either bonded or nonbonded

    """
    obj_dict = ast.literal_eval(obj)

    handlers = []

    protein_ff = obj_dict.pop(serialization_format.PROTEIN_FF_TAG, constants.DEFAULT_PROTEIN_FF)
    water_ff = obj_dict.pop(serialization_format.WATER_FF_TAG, constants.DEFAULT_WATER_FF)

    for k, v in obj_dict.items():
        cls_name = k + _SUFFIX

        ctor = None

        try:
            ctor = getattr(bonded, cls_name)
        except AttributeError:
            pass

        try:
            ctor = getattr(nonbonded, cls_name)
        except AttributeError:
            pass

        if ctor is None:
            raise Exception("Unknown handler:", k)

        patterns = v["patterns"]
        smirks = []
        params = []

        for elems in patterns:
            smirks.append(elems[0])
            if len(elems) == 2:
                params.append(elems[1])
            else:
                params.append(elems[1:])

        props = v.get("props")

        handlers.append(ctor(smirks, params, props))

    return handlers, protein_ff, water_ff
