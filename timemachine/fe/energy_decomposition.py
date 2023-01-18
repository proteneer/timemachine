import functools
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from timemachine.constants import BOLTZ, DEFAULT_TEMP


@dataclass
class EnergyDecomposedState:
    frames: np.ndarray
    boxes: np.ndarray
    batch_u_fns: Sequence[Callable]  # u_fn : (frames, boxes) -> reduced_energies


def get_batch_u_fns(bps, temperature=DEFAULT_TEMP):
    # return a list of functions that take in (coords, boxes), return reduced_potentials

    kBT = temperature * BOLTZ

    batch_u_fns = []
    for bp in bps:

        def batch_u_fn(xs, boxes, bp_impl):
            Us = []
            for x, box in zip(xs, boxes):
                # tbd optimize to "selective" later
                _, U = bp_impl.execute(x, box)
                Us.append(U)
            us = np.array(Us) / kBT
            return us

        # extra functools.partial is needed to deal with closure jank
        batch_u_fns.append(functools.partial(batch_u_fn, bp_impl=bp))

    return batch_u_fns


# TODO: generalize to list of >= 2 states


def compute_energy_decomposed_u_kln(state_0: EnergyDecomposedState, state_1: EnergyDecomposedState) -> np.ndarray:
    """

    Parameters
    ----------
    state_0, state_1: EnergyDecomposedStates
        each contains samples (frames, boxes) and a list of energy functions

    Returns
    -------
    u_kln_by_component : [n_components, 2, 2, n_frames]

        u_kln_by_component[comp, k, l, n] =
            state k energy component comp,
            evaluated on sample n from state l
    """

    n_frames = state_0.frames.shape[0]
    n_components = len(state_0.batch_u_fns)

    assert state_1.frames.shape[0] == n_frames
    assert len(state_1.batch_u_fns) == n_components

    states = [state_0, state_1]
    u_kln_by_component = np.zeros((n_components, 2, 2, n_frames))
    for comp in range(n_components):
        for src in range(2):
            xs, boxes = states[src].frames, states[src].boxes
            for dst in range(2):
                u_fxn = states[dst].batch_u_fns[comp]

                u_kln_by_component[comp, src, dst] = u_fxn(xs, boxes)

    return u_kln_by_component
