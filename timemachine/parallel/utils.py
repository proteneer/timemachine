from subprocess import check_output
from timemachine.parallel.grpc.service_pb2 import StatusResponse


def get_gpu_count() -> int:
    output = check_output(["nvidia-smi", "-L"])
    # Expected to return a line delimited summary of each GPU
    return len([x for x in output.split(b"\n") if len(x)])


def get_worker_status() -> StatusResponse:
    try:
        with open("/proc/driver/nvidia/version") as ifs:
            nvidia_driver = ifs.read().strip()
    except FileNotFoundError:
        nvidia_driver = ""
    try:
        git_sha = check_output(["git", "rev-parse", "HEAD"]).strip()
    except FileNotFoundError:
        git_sha = ""
    return StatusResponse(
        nvidia_driver=nvidia_driver,
        git_sha=git_sha,
    )
