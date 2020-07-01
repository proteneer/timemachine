import ast
import numpy as np

from ff.handlers import bonded, nonbonded
from ff.handlers.suffix import _SUFFIX

def deserialize(obj):
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

    for k, v in obj_dict.items():

        cls_name = k+_SUFFIX

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

        patterns = v['patterns']
        smirks = []
        params = []

        for elems in patterns:
            smirks.append(elems[0])
            params.append(elems[1:])

        params = np.array(params, dtype=np.float64)
        params = np.squeeze(params) # remove single dimension entries, eg. charge params

        props = v.get('props')

        handlers.append(ctor(smirks, params, props))

    return handlers
