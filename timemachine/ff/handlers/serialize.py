import io
import pprint

import numpy as np

from timemachine.ff.handlers import serialization_format
from timemachine.ff.handlers.suffix import _SUFFIX


def serialize_handlers(all_handlers, protein_ff, water_ff):
    final_ff = {}
    final_ff[serialization_format.PROTEIN_FF_TAG] = protein_ff
    final_ff[serialization_format.WATER_FF_TAG] = water_ff

    for handler in all_handlers:
        if handler is None:  # optional handler not specified
            continue
        ff_obj = handler.serialize()

        for k in ff_obj.keys():
            assert k not in final_ff, f"Handler {k} already exists"

        final_ff.update(ff_obj)

    return bin_to_str(final_ff)


def bin_to_str(binary):
    buf = io.StringIO()
    pp = pprint.PrettyPrinter(width=500, compact=False, stream=buf)
    pp._sorted = lambda x: x
    pp.pprint(binary)
    return buf.getvalue()


class SerializableMixIn:
    def serialize(self):
        """

        Returns
        -------
        result : dict
        """
        handler = self
        key = type(handler).__name__[: -len(_SUFFIX)]
        patterns = []
        for smi, p in zip(handler.smirks, handler.params):
            if isinstance(p, (list, tuple)):
                patterns.append((smi, *p))
            elif isinstance(p, np.ndarray):
                patterns.append((smi, *p.tolist()))
            else:
                # SimpleCharges only have one parameter
                patterns.append((smi, p))

        body = {"patterns": patterns}
        if handler.props is not None:
            body["props"] = handler.props

        result = {key: body}

        return result
