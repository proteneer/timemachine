from typing import List, Tuple, Generator


FrameIterator = Generator[Tuple[int, "SimulationResult"], None, None]


def all_frames(results: List["SimulationResult"]) -> FrameIterator:
    return enumerate(results)


def endpoint_frames_only(results: List["SimulationResult"]) -> FrameIterator:
    output = []
    if len(results) == 0:
        return output
    output.append((0, results[0]))
    # In the case of only one, we don't want to duplicate the frame
    if len(results) > 1:
        output.append((len(results) - 1, results[-1]))
    return output


def no_frames(results: List["SimulationResult"]) -> FrameIterator:
    return []
