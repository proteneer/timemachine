import os
from glob import glob
from pathlib import Path
from warnings import catch_warnings

import pytest
from common import temporary_working_dir

from timemachine.ff import Forcefield
from timemachine.ff.handlers.deserialize import deserialize_handlers


def test_serialization_of_ffs():
    for path in glob("timemachine/ff/params/smirnoff_*.py"):
        ff = Forcefield(deserialize_handlers(open(path).read()))
        for handle in ff.get_ordered_handles():
            assert handle is not None, f"{path} failed to deserialize correctly"


def test_loading_forcefield_from_file():
    builtin_ffs = glob("timemachine/ff/params/smirnoff_*.py")
    for path in builtin_ffs:
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

    with temporary_working_dir():
        # Verify that if a local file shadows a builtin
        for path in builtin_ffs:
            basename = os.path.basename(path)
            with open(basename, "w") as ofs:
                ofs.write("junk")
            with catch_warnings(record=True) as w:
                Forcefield.load_from_file(basename)
            assert len(w) == 1
            assert basename in str(w[0].message)
