import ast
import io
import pprint
import numpy as np

from ff.handlers import bonded, nonbonded

_SUFFIX = "Handler"

def serialize(handler):
    """
    Parameters
    ----------
    handler: instance of a handler
        The handler we will be serialization

    Returns
    -------
    str
        serialized string representation.

    """
    key = type(handler).__name__[:-len(_SUFFIX)]
    patterns = []
    for smi, p in zip(handler.smirks, handler.params):
        if isinstance(p, (list, tuple, np.ndarray)):
            patterns.append((smi, *p))
        else:
            # SimpleCharges only have one parameter
            patterns.append((smi, p))

    body = {'patterns': patterns}
    if handler.props is not None:
        body['props'] = handler.props

    result = {key: body}

    buf = io.StringIO()
    pp = pprint.PrettyPrinter(width=500, compact=False, stream=buf)
    pp._sorted = lambda x:x
    pp.pprint(result)

    return buf.getvalue()

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
