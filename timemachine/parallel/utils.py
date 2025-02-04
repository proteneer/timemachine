import os
from collections import defaultdict
from subprocess import check_output
from typing import Optional


def get_gpu_count() -> int:
    # Expected to return a line delimited summary of each GPU
    try:
        output = check_output(["nvidia-smi", "-L"])
    except FileNotFoundError:
        return 0
    gpu_list = [x for x in output.split(b"\n") if len(x)]

    # Respect CUDA_VISIBLE_DEVICES in determining GPU count
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        gpu_list = [gpu_list[i] for i in map(int, visible_devices.split(","))]

    return len(gpu_list)


def batch_list(values: list, num_workers: Optional[int] = None) -> list[list]:
    """
    Split a list of values into `num_workers` batches.
    If num_workers is None, then split each value into a separate batch.
    """
    batched_values = defaultdict(list)
    num_workers = num_workers or len(values)
    for i, value in enumerate(values):
        batched_values[i % num_workers].append(value)
    return list(batched_values.values())
