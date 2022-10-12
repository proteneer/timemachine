from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Iterable, Optional, Tuple, TypeVar, Union
from warnings import warn

import importlib_resources as resources

from timemachine.constants import DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF
from timemachine.ff.handlers import bonded, nonbonded
from timemachine.ff.handlers.deserialize import deserialize_handlers
from timemachine.ff.handlers.serialize import serialize_handlers

_T = TypeVar("_T")


@dataclass
class ForcefieldParams(Generic[_T]):
    hb_params: _T
    ha_params: _T
    pt_params: _T
    it_params: _T
    q_params: _T
    lj_params: _T


def combine_params(a: ForcefieldParams[_T], b: ForcefieldParams[_T]) -> ForcefieldParams[Tuple[_T, _T]]:
    return ForcefieldParams(
        (a.hb_params, b.hb_params),
        (a.ha_params, b.ha_params),
        (a.pt_params, b.pt_params),
        (a.it_params, b.it_params),
        (a.q_params, b.q_params),
        (a.lj_params, b.lj_params),
    )


@dataclass(frozen=True)
class Forcefield:
    """
    Utility class for wrapping around a list of ff_handlers
    """

    hb_handle: Optional[bonded.HarmonicBondHandler]
    ha_handle: Optional[bonded.HarmonicAngleHandler]
    pt_handle: Optional[bonded.ProperTorsionHandler]
    it_handle: Optional[bonded.ImproperTorsionHandler]
    q_handle: Optional[Union[nonbonded.SimpleChargeHandler, nonbonded.AM1BCCHandler, nonbonded.AM1CCCHandler]]
    lj_handle: Optional[nonbonded.LennardJonesHandler]

    protein_ff: str
    water_ff: str

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
                handlers, protein_ff, water_ff = deserialize_handlers(ifs.read())

        return cls.from_handlers(handlers, protein_ff=protein_ff, water_ff=water_ff)

    @classmethod
    def from_handlers(
        cls,
        ff_handlers: Iterable[Any],
        protein_ff: str = DEFAULT_PROTEIN_FF,
        water_ff: str = DEFAULT_WATER_FF,
    ):
        hb_handle = None
        ha_handle = None
        pt_handle = None
        it_handle = None
        lj_handle = None
        q_handle = None

        for handle in ff_handlers:
            if isinstance(handle, bonded.HarmonicBondHandler):
                assert hb_handle is None
                hb_handle = handle
            elif isinstance(handle, bonded.HarmonicAngleHandler):
                assert ha_handle is None
                ha_handle = handle
            elif isinstance(handle, bonded.ProperTorsionHandler):
                assert pt_handle is None
                pt_handle = handle
            elif isinstance(handle, bonded.ImproperTorsionHandler):
                assert it_handle is None
                it_handle = handle
            elif isinstance(handle, nonbonded.LennardJonesHandler):
                assert lj_handle is None
                lj_handle = handle
            elif (
                isinstance(handle, nonbonded.AM1CCCHandler)
                or isinstance(handle, nonbonded.AM1BCCHandler)
                or isinstance(handle, nonbonded.SimpleChargeHandler)
            ):
                assert q_handle is None
                q_handle = handle

        return cls(hb_handle, ha_handle, pt_handle, it_handle, q_handle, lj_handle, protein_ff, water_ff)

    def get_ordered_handles(self):
        """Returns a list of handlers with deterministic ordering."""
        return [self.hb_handle, self.ha_handle, self.pt_handle, self.it_handle, self.q_handle, self.lj_handle]

    def get_params(self) -> ForcefieldParams:
        def params(x):
            return x.params if x is not None else None

        return ForcefieldParams(
            params(self.hb_handle),
            params(self.ha_handle),
            params(self.pt_handle),
            params(self.it_handle),
            params(self.q_handle),
            params(self.lj_handle),
        )

    def serialize(self) -> str:
        return serialize_handlers(self.get_ordered_handles(), protein_ff=self.protein_ff, water_ff=self.water_ff)


def sanitize_water_ff(water_ff: str) -> str:
    """
    Return the sanitized water name for the given water_ff.

    For example tip3pfb -> tip3p.
    """
    water_ff = water_ff.split("/")[-1]
    # Use consistent model name for the various water flavors
    if water_ff.lower() in ["tip3p", "tip3pfb"]:
        return "tip3p"
    if water_ff.lower() in ["tip4p", "tip4pew", "tip4pfb"]:
        return "tip4p"
    return water_ff
