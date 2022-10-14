import multiprocessing
import os
import pickle
from concurrent import futures
from pathlib import Path
from typing import Any, List, Optional, Union

import grpc
from timemachine.parallel.grpc import service_pb2, service_pb2_grpc
from timemachine.parallel.utils import get_gpu_count

DEFAULT_GRPC_OPTIONS = [
    ("grpc.max_send_message_length", 1024 * 1024 * 1024),
    ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
]

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


class GRPCClient(AbstractClient):
    def __init__(self, hosts: Union[str, List[str]], options: Optional[List[Any]] = None, default_port: int = 8888):
        """
        GRPCClient is meant for distributed use. The GRPC workers must
        be launched prior to starting the client. The worker version should be compatible
        with that of client, as pickling requires that the files are consistent
        across the client and the server.

        Parameters
        ----------
        hosts: str or list
            List of hosts to use as GRPC workers or a line delimited file with hosts.
        options: list
            List of options to configure GRPC connections
        default_port: int
            Default port to expect client running on, if none provided

        """
        self.hosts = self._prepare_hosts(hosts, default_port)
        self.stubs = []
        if options is None:
            options = DEFAULT_GRPC_OPTIONS
        for host in self.hosts:
            channel = grpc.insecure_channel(
                host,
                options=options,
            )
            self.stubs.append(service_pb2_grpc.WorkerStub(channel))
        self._idx = 0
        self._total_idx = 0
        self.max_workers = len(self.hosts)

    def _prepare_hosts(self, hosts: Union[str, List[str]], default_port: int):
        if isinstance(hosts, str):
            assert os.path.isfile(hosts), f"{hosts} is not a file or a list of hosts"
            new_hosts = []
            with open(hosts, "r") as ifs:
                for line in ifs.readlines():
                    new_hosts.append(line.strip())
            hosts = new_hosts
        modded_hosts = []
        for host in hosts:
            if ":" not in host:
                host = f"{host}:{default_port}"
            modded_hosts.append(host)
        return modded_hosts

    def submit(self, task_fn, *args, **kwargs):
        """
        See abstract class for documentation.
        """
        binary = pickle.dumps((task_fn, args, kwargs))
        request = service_pb2.PickleData(binary=binary)
        future = self.stubs[self._idx].Submit.future(request)
        job_id = str(self._total_idx)
        self._total_idx += 1
        self._idx = (self._idx + 1) % len(self.stubs)
        return BinaryFutureWrapper(future, job_id)

    def verify(self):
        """
        See abstract class for documentation.
        """
        prev_vals = {}
        for host, stub in zip(self.hosts, self.stubs):
            try:
                status = stub.Status(service_pb2.StatusRequest())
            except grpc.RpcError as e:
                raise AssertionError(f"Failed to connect to {host}") from e
            for field in (
                "nvidia_driver",
                "git_sha",
            ):
                # All fields should be the same
                new_val = getattr(status, field)
                if field in prev_vals and prev_vals[field] != new_val:
                    assert False, f"Host {host} '{field}' new value of {new_val} != {prev_vals[field]}"
                else:
                    prev_vals[field] = new_val
