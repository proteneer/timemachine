import os
from subprocess import check_output


def get_gpu_count() -> int:
    output = check_output(["nvidia-smi", "-L"])
    # Expected to return a line delimited summary of each GPU
    return len([x for x in output.split(b"\n") if len(x)])

