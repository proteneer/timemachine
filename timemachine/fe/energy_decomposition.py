import functools
from dataclasses import dataclass
from typing import Callable, List, Sequence

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


def compute_energy_decomposed_u_kln(states: List[EnergyDecomposedState]) -> np.ndarray:
    """

    Parameters
    ----------
    states: [K] list of EnergyDecomposedStates
        each contains samples (frames, boxes) and a list of energy functions

    Returns
    -------
    u_kln_by_component : [n_components, K, K, n_frames]

        u_kln_by_component[comp, k, l, n] =
            state k energy component comp,
            evaluated on sample n from state l
    """

    K = len(states)
    n_frames = states[0].frames.shape[0]
    n_components = len(states[0].batch_u_fns)

    for state in states:
        assert state.frames.shape[0] == n_frames
        assert len(state.batch_u_fns) == n_components

    u_kln_by_component = np.zeros((n_components, K, K, n_frames))
    for comp in range(n_components):
        for k in range(K):
            u_fxn = states[k].batch_u_fns[comp]
            for l in range(K):
                xs, boxes = states[l].frames, states[l].boxes
                u_kln_by_component[comp, k, l] = u_fxn(xs, boxes)

    return u_kln_by_component
