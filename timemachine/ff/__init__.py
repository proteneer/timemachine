from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar, Union
from warnings import warn

from numpy.testing import assert_equal

from timemachine.constants import DEFAULT_FF, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF
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
    q_params_intra: _T
    lj_params: _T
    lj_params_intra: _T


def combine_params(a: ForcefieldParams[_T], b: ForcefieldParams[_T]) -> ForcefieldParams[tuple[_T, _T]]:
    return ForcefieldParams(
        (a.hb_params, b.hb_params),
        (a.ha_params, b.ha_params),
        (a.pt_params, b.pt_params),
        (a.it_params, b.it_params),
        (a.q_params, b.q_params),
        (a.q_params_intra, b.q_params_intra),
        (a.lj_params, b.lj_params),
        (a.lj_params_intra, b.lj_params_intra),
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
    q_handle: Optional[
        Union[
            nonbonded.SimpleChargeHandler,
            nonbonded.AM1BCCHandler,
            nonbonded.RESPHandler,
            nonbonded.AM1CCCHandler,
            nonbonded.AM1BCCCCCHandler,
            nonbonded.PrecomputedChargeHandler,
            nonbonded.NNHandler,
        ]
    ]
    q_handle_intra: Optional[
        Union[
            nonbonded.SimpleChargeIntraHandler,
            nonbonded.AM1BCCIntraHandler,
            nonbonded.AM1CCCIntraHandler,
            nonbonded.AM1BCCCCCIntraHandler,
            nonbonded.PrecomputedChargeHandler,
            nonbonded.NNHandler,
        ]
    ]

    lj_handle: Optional[nonbonded.LennardJonesHandler]
    lj_handle_intra: Optional[nonbonded.LennardJonesIntraHandler]
    env_bcc_handle: Optional[
        Union[
            nonbonded.EnvironmentBCCPartialHandler,
            nonbonded.EnvironmentNNPartialHandler,
        ]
    ]

    protein_ff: str
    water_ff: str

    @classmethod
    def load_from_file(cls, path_or_str: Union[str, Path]) -> "Forcefield":
        """Load a forcefield from a path

        Parameters
        ----------
        path_or_str: string or pathlib.Path
            Either the filename of a built in ff (smirnoff_1_1_0_sc.py) or a path to a new forcefield file

        Returns
        -------
        Forcefield
            Return a Forcefield object constructed from parameters file

        Note
        ----
        If a path is provided that has the same file name as a built-in forcefield, it will throw a warning and load
        the built-in forcefield.
        """
        original_path = str(path_or_str)
        path = Path(path_or_str)  # Safe to construct a Path object from another Path object

        # Look for builtin ff
        with resources.as_file(resources.files("timemachine.ff.params") / path.name) as rpath:
            if rpath.exists():
                if path.is_file():
                    warn(
                        f"Provided path {original_path} shares name with built-in forcefield, falling back to built-in"
                    )
                handlers, protein_ff, water_ff = deserialize_handlers(rpath.read_text())
                return cls.from_handlers(handlers, protein_ff=protein_ff, water_ff=water_ff)

        # Look for ff file
        if not path.is_file():
            raise ValueError(f"Unable to find {original_path} in file system or built-in forcefields")

        handlers, protein_ff, water_ff = deserialize_handlers(path.read_text())
        return cls.from_handlers(handlers, protein_ff=protein_ff, water_ff=water_ff)

    @classmethod
    def load_default(cls) -> "Forcefield":
        """alias for load_from_file(DEFAULT_FF)"""
        return cls.load_from_file(DEFAULT_FF)

    @classmethod
    def load_precomputed_default(cls) -> "Forcefield":
        """load a default forcefield where charges are read in from the Molblock"""
        ff = cls.load_default()
        q_handle = nonbonded.PrecomputedChargeHandler([], [], None)
        q_handle_intra = nonbonded.PrecomputedChargeIntraHandler([], [], None)
        return Forcefield(
            ff.hb_handle,
            ff.ha_handle,
            ff.pt_handle,
            ff.it_handle,
            q_handle=q_handle,
            q_handle_intra=q_handle_intra,
            lj_handle=ff.lj_handle,
            lj_handle_intra=ff.lj_handle_intra,
            env_bcc_handle=None,
            protein_ff=ff.protein_ff,
            water_ff=ff.water_ff,
        )

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
        lj_handle_intra = None
        lj_handle_solv = None
        q_handle = None
        q_handle_intra = None
        q_handle_solv = None
        env_bcc_handle = None

        for handle in ff_handlers:
            if isinstance(
                handle,
                (nonbonded.EnvironmentBCCPartialHandler, nonbonded.EnvironmentNNPartialHandler),
            ):
                assert env_bcc_handle is None
                env_bcc_handle = handle
            elif isinstance(handle, bonded.HarmonicBondHandler):
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
            elif isinstance(handle, nonbonded.LennardJonesIntraHandler):
                assert lj_handle_intra is None
                lj_handle_intra = handle
            elif isinstance(handle, nonbonded.LennardJonesSolventHandler):
                assert lj_handle_solv is None
                lj_handle_solv = handle
            elif isinstance(handle, nonbonded.LennardJonesHandler):
                assert lj_handle is None
                lj_handle = handle
            elif isinstance(
                handle,
                (
                    nonbonded.AM1BCCCCCIntraHandler,
                    nonbonded.AM1CCCIntraHandler,
                    nonbonded.AM1BCCIntraHandler,
                    nonbonded.SimpleChargeIntraHandler,
                    nonbonded.PrecomputedChargeIntraHandler,
                ),
            ):
                # Need to be checked first since they are also subclasses
                # of the non-intra handlers
                assert q_handle_intra is None
                q_handle_intra = handle
            elif isinstance(
                handle,
                (nonbonded.AM1CCCSolventHandler, nonbonded.AM1BCCSolventHandler, nonbonded.SimpleChargeSolventHandler),
            ):
                assert q_handle_solv is None
                q_handle_solv = handle
            elif isinstance(
                handle,
                (
                    nonbonded.AM1BCCCCCHandler,
                    nonbonded.AM1CCCHandler,
                    nonbonded.AM1BCCHandler,
                    nonbonded.SimpleChargeHandler,
                    nonbonded.PrecomputedChargeHandler,
                    nonbonded.NNHandler,
                ),
            ):
                assert q_handle is None
                q_handle = handle

        if lj_handle_intra is None:
            if isinstance(lj_handle, nonbonded.LennardJonesHandler):
                lj_handle_intra = nonbonded.LennardJonesIntraHandler(
                    lj_handle.smirks, lj_handle.params, lj_handle.props
                )

        if lj_handle_solv is not None:
            assert lj_handle is not None
            assert lj_handle_solv is not None
            assert_equal(
                lj_handle.params,
                lj_handle_solv.params,
                err_msg="Split ligand-solvent LJ interactions are no longer supported",
            )

        if q_handle_intra is None and q_handle is not None:
            # Copy the forcefield parameters to the intramolecular term if not
            # already handled.
            if isinstance(q_handle, nonbonded.AM1BCCCCCHandler):
                q_handle_intra = nonbonded.AM1BCCCCCIntraHandler(q_handle.smirks, q_handle.params, q_handle.props)
            elif isinstance(q_handle, nonbonded.AM1CCCHandler):
                q_handle_intra = nonbonded.AM1CCCIntraHandler(q_handle.smirks, q_handle.params, q_handle.props)
            elif isinstance(q_handle, nonbonded.AM1BCCHandler):
                q_handle_intra = nonbonded.AM1BCCIntraHandler(q_handle.smirks, q_handle.params, q_handle.props)
            elif isinstance(q_handle, nonbonded.RESPHandler):
                q_handle_intra = nonbonded.RESPHandler(q_handle.smirks, q_handle.params, q_handle.props)
            elif isinstance(q_handle, nonbonded.SimpleChargeHandler):
                q_handle_intra = nonbonded.SimpleChargeIntraHandler(q_handle.smirks, q_handle.params, q_handle.props)
            elif isinstance(q_handle, nonbonded.PrecomputedChargeHandler):
                q_handle_intra = nonbonded.PrecomputedChargeIntraHandler(
                    q_handle.smirks, q_handle.params, q_handle.props
                )
            else:
                raise ValueError(f"Unsupported charge handler {q_handle}")

        if q_handle_solv is not None:
            assert q_handle is not None
            assert q_handle_solv is not None
            assert_equal(
                q_handle.params,
                q_handle_solv.params,
                err_msg="Split ligand-solvent charge interactions are no longer supported",
            )

        return cls(
            hb_handle,
            ha_handle,
            pt_handle,
            it_handle,
            q_handle,
            q_handle_intra,
            lj_handle,
            lj_handle_intra,
            env_bcc_handle,
            protein_ff,
            water_ff,
        )

    def get_ordered_handles(self):
        """Returns a list of handlers with deterministic ordering."""
        return [
            self.hb_handle,
            self.ha_handle,
            self.pt_handle,
            self.it_handle,
            self.q_handle,
            self.q_handle_intra,
            self.lj_handle,
            self.lj_handle_intra,
            self.env_bcc_handle,
        ]

    def get_params(self) -> ForcefieldParams:
        def params(x):
            return x.params if x is not None else None

        return ForcefieldParams(
            params(self.hb_handle),
            params(self.ha_handle),
            params(self.pt_handle),
            params(self.it_handle),
            params(self.q_handle),
            params(self.q_handle_intra),
            params(self.lj_handle),
            params(self.lj_handle_intra),
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
        return "tip4pew"
    return water_ff
