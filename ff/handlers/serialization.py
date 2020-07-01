import ast
import io
import pprint

from ff.handlers import bonded, nonbonded

_SUFFIX = "Handler"

# def save(self, handle):
#     with open(handle, "w") as fh:
#         pp = pprint.PrettyPrinter(width=500, compact=False, stream=fh)
#         pp._sorted = lambda x:x
#         pp.pprint(self.serialize())

def serialize(handler):
    """
    Parameters
    ----------
    handler: instance of a handler
        The handler we will be serialization

    Returns
    -------
    A python object can be the turned into a string format.

    """
    key = type(handler).__name__[:-len(_SUFFIX)]
    patterns = []
    for smi, p in zip(handler.smirks, handler.params):
        patterns.append((smi, *p))

    result = {key: {'patterns': patterns}}

    buf = io.StringIO()
    pp = pprint.PrettyPrinter(width=500, compact=False, stream=buf)
    pp._sorted = lambda x:x
    pp.pprint(result)

    return buf.getvalue()
    # return result

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

    # if isinstance(handle, str):
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


        handlers.append(ctor(smirks, params))

    return handlers
