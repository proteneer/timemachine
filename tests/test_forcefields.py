from glob import glob
from timemachine.ff import Forcefield
from timemachine.ff.handlers.deserialize import deserialize_handlers


def test_serialization_of_ffs():
    for path in glob("timemachine/ff/params/smirnoff_*.py"):
        ff = Forcefield(deserialize_handlers(open(path).read()))
        for handle in ff.get_ordered_handles():
            assert handle is not None, f"{path} failed to deserialize correctly"
