from pathlib import Path
from typing import Union
from warnings import warn

import importlib_resources as resources

from timemachine.ff.handlers import bonded, nonbonded
from timemachine.ff.handlers.deserialize import deserialize_handlers


class Forcefield:
    """
    Utility class for wrapping around a list of ff_handlers
    """

    @classmethod
    def load_from_file(cls, path_or_str: Union[str, Path]) -> "Forcefield":
        """Load a forcefield from a path

        Parameters
        ----------

        path: string or pathlib.Path
            Either the filename of a built in ff (smirnoff_1_1_0_sc.py) or a path to a new forcefield file

        Returns
        -------
        Forcefield
            Return a ForceField object constructed from parameters file
        """
        original_path = str(path_or_str)
        path = Path(path_or_str)  # Safe to construct a Path object from another Path object

        with resources.files("timemachine.ff.params") as params_path:
            built_in_path = params_path / original_path
            if built_in_path.is_file():
                if path.is_file():
                    warn(f"Provided path {original_path} shares name with builtin forcefield, falling back to builtin")
                # Search built in params for the forcefields
                path = built_in_path
            if not path.is_file():
                raise ValueError(f"Unable to find {original_path} in file system or built in forcefields")
            with open(path, "r") as ifs:
                handlers = deserialize_handlers(ifs.read())

        return cls(handlers)

    def __init__(self, ff_handlers):
        self.hb_handle = None
        self.ha_handle = None
        self.pt_handle = None
        self.it_handle = None
        self.lj_handle = None
        self.q_handle = None
        for handle in ff_handlers:
            if isinstance(handle, bonded.HarmonicBondHandler):
                assert self.hb_handle is None
                self.hb_handle = handle
            if isinstance(handle, bonded.HarmonicAngleHandler):
                assert self.ha_handle is None
                self.ha_handle = handle
            if isinstance(handle, bonded.ProperTorsionHandler):
                assert self.pt_handle is None
                self.pt_handle = handle
            if isinstance(handle, bonded.ImproperTorsionHandler):
                assert self.it_handle is None
                self.it_handle = handle
            if isinstance(handle, nonbonded.LennardJonesHandler):
                assert self.lj_handle is None
                self.lj_handle = handle
            if isinstance(handle, nonbonded.AM1CCCHandler):
                assert self.q_handle is None
                self.q_handle = handle
            if isinstance(handle, nonbonded.AM1BCCHandler):
                assert self.q_handle is None
                self.q_handle = handle
            if isinstance(handle, nonbonded.SimpleChargeHandler):
                assert self.q_handle is None
                self.q_handle = handle

    def get_ordered_params(self):
        """
        Returns
        -------
        list of np.ndarray
            Return a flat, pre-determined ordering of the parameters
        """
        return [x.params for x in self.get_ordered_handles()]

    def get_ordered_handles(self):
        """
        Returns
        -------
        list of np.ndarray
            Return a flat, pre-determined ordering of the handlers
        """
        return [self.hb_handle, self.ha_handle, self.pt_handle, self.it_handle, self.q_handle, self.lj_handle]


def combine_ordered_params(ff0: Forcefield, ff1: Forcefield):
    return list(zip(ff0.get_ordered_params(), ff1.get_ordered_params()))
