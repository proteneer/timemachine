import os
from collections import defaultdict
from subprocess import check_output
from typing import List

from timemachine.parallel.grpc.service_pb2 import StatusResponse


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


def get_worker_status() -> StatusResponse:  # type: ignore
    try:
        with open("/proc/driver/nvidia/version") as ifs:
            nvidia_driver = ifs.read().strip()
    except FileNotFoundError:
        nvidia_driver = ""
    try:
        git_sha = check_output(["git", "rev-parse", "HEAD"]).strip()
    except FileNotFoundError:
        git_sha = b""
    return StatusResponse(
        nvidia_driver=nvidia_driver,
        git_sha=git_sha,
    )


def batch_list(values: List, num_workers: int) -> List[List]:
    batched_values = defaultdict(list)
    for i, value in enumerate(values):
        batched_values[i % num_workers].append(value)
    return list(batched_values.values())


def flatten_list(results: List[List[str]]) -> List[str]:
    return [i for j in results for i in j]
