import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from timemachine.constants import BOLTZ, DEFAULT_TEMP
from timemachine.lib.custom_ops import Potential
from timemachine.potentials.types import Params

Frames = TypeVar("Frames")
Boxes = list[NDArray]
ReducedEnergies = np.ndarray
Batch_u_fn = Callable[[Frames, Boxes], ReducedEnergies]


@dataclass
class EnergyDecomposedState(Generic[Frames]):
    """contains samples (frames, boxes) and a list of reduced energy functions"""

    frames: Frames
    boxes: Boxes
    batch_u_fns: Sequence[Batch_u_fn]  # u_fn : (frames, boxes) -> reduced_energies


def get_batch_u_fns(
    pots: Sequence[Potential],
    params: Sequence[Params],
    temperature: float = DEFAULT_TEMP,
) -> list[Batch_u_fn]:
    """Get a list of functions that take in (coords, boxes), return reduced_potentials

    Parameters
    ----------
    pots: list of potential impls
    params: list of parameters to call potentials on
    temperature: float

    Returns
    -------
    list of Batch_u_fn 's
    """
    kBT = temperature * BOLTZ

    assert len(pots) == len(params)
    batch_u_fns: list[Batch_u_fn] = []
    for p, pot in zip(params, pots):

        def batch_u_fn(xs: Frames, boxes: Boxes, pot_impl, pot_params) -> ReducedEnergies:
            # If the coordinates are already an in-memory numpy array, don't create a copy
            coords = np.asarray(xs)
            _, _, Us = pot_impl.execute_batch(
                coords,
                pot_params,
                np.asarray(boxes),
                compute_du_dx=False,
                compute_du_dp=False,
                compute_u=True,
            )
            # ravel the array to remove the params dimension
            us = Us.ravel() / kBT
            return us

        # extra functools.partial is needed to deal with closure jank
        batch_u_fns.append(functools.partial(batch_u_fn, pot_impl=pot, pot_params=p[np.newaxis]))

    return batch_u_fns


def compute_energy_decomposed_u_kln(states: list[EnergyDecomposedState]) -> np.ndarray:
    """Compute a stack of u_kln matrices, one per energy component

    Parameters
    ----------
    states: [K] list of EnergyDecomposedStates
        each contains samples (frames, boxes) and a list of energy functions

    Returns
    -------
    u_kln_by_component : [n_components, K, K, n_frames]

        u_kln_by_component[comp, k, l, n] =
            sample n from state k
            evaulated using the energy function l
            (PyMBAR convention)
    """

    K = len(states)
    n_frames = len(states[0].frames)
    n_components = len(states[0].batch_u_fns)

    for state in states:
        assert len(state.frames) == n_frames
        assert len(state.batch_u_fns) == n_components

    u_kln_by_component = np.zeros((n_components, K, K, n_frames))
    for k in range(K):
        # Load the frames into memory, then evaluate all of the components
        # Done to avoid repeatedly reading from disk
        xs, boxes = np.array(states[k].frames), states[k].boxes
        for l in range(K):
            for comp in range(n_components):
                u_fxn = states[l].batch_u_fns[comp]
                u_kln_by_component[comp, k, l] = u_fxn(xs, boxes)

    return u_kln_by_component
