import io
import pprint
import numpy as np

from ff.handlers.suffix import _SUFFIX

class SerializableMixIn():

    def serialize(self):
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

        buf = io.StringIO()
        pp = pprint.PrettyPrinter(width=500, compact=False, stream=buf)
        pp._sorted = lambda x:x
        pp.pprint(result)

        return buf.getvalue()
