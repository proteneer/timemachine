import io
import pprint
import numpy as np

from ff.handlers.suffix import _SUFFIX

def serialize_handlers(all_handlers):

    final_ff = {}

    for handler in all_handlers:
        ff_obj = handler.serialize()

        for k in ff_obj.keys():
            assert k not in final_ff

        final_ff.update(ff_obj)

    return bin_to_str(final_ff)


def bin_to_str(binary):
    buf = io.StringIO()
    pp = pprint.PrettyPrinter(width=500, compact=False, stream=buf)
    pp._sorted = lambda x:x
    pp.pprint(binary)
    return buf.getvalue()

class SerializableMixIn():

    def serialize(self):
        """

        Returns
        -------
        result : dict
        """
        handler = self
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
        
        return result
