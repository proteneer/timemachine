import os
from glob import glob
from pathlib import Path

import pytest

from timemachine.ff import Forcefield
from timemachine.ff.handlers.deserialize import deserialize_handlers


def test_serialization_of_ffs():
    for path in glob("timemachine/ff/params/smirnoff_*.py"):
        ff = Forcefield(deserialize_handlers(open(path).read()))
        for handle in ff.get_ordered_handles():
            assert handle is not None, f"{path} failed to deserialize correctly"


def test_loading_forcefield_from_file():
    for path in glob("timemachine/ff/params/smirnoff_*.py"):
        # Use the full path
        ff = Forcefield.load_from_file(path)
        assert ff is not None
        # Use full path as Path object
        ff = Forcefield.load_from_file(Path(path))
        assert ff is not None
        # Load using just file name of the built in
        ff = Forcefield.load_from_file(os.path.basename(path))
        assert ff is not None

    for prefix in ["", "timemachine/ff/params/"]:
        path = os.path.join(prefix, "nosuchfile.py")
        with pytest.raises(ValueError) as e:
            Forcefield.load_from_file(path)
        assert path in str(e.value)
        with pytest.raises(ValueError):
            Forcefield.load_from_file(Path(path))
        assert path in str(e.value)
