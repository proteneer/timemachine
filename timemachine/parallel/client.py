import multiprocessing
import os
import pickle
from concurrent import futures
from pathlib import Path
from typing import Any, List, Optional, Union

import grpc
from timemachine.parallel.constants import DEFAULT_GRPC_OPTIONS
from timemachine.parallel.grpc import service_pb2, service_pb2_grpc
from timemachine.parallel.utils import get_gpu_count

# (ytz): The classes in this file are designed to help provide a consistent API between
# multiprocessing (typically for local cluster use) and gRPC (distributed and multi-node).


class AbstractClient:
    def submit(self, task_fn, *args, **kwargs):
        """
        Submit is an asynchronous method that will launch task_fn whose
        results will be collected at a later point in time. The input task_fn
        and its arguments should be picklable. See Python documentation for
        pickle rules.

        Parameters
        ----------
        task_fn: callable
            A python function to be called

        args: list
            list of arguments for task_fn


        Returns
        -------
        Future
            A deferred object with a .result() method.

        Usage:

        client = ConcreteClient()

        futures = []
        for arg in args:
            fut = client.submit(task_fn, arg)
            futures.append(fut)

        res = []
        for fut in futures:
            res.append(fut.result())

        """
        raise NotImplementedError()

    def verify(self):
        """Verify performs any necessary checks to verify the client is ready to
        handle calls to submit.

        Raises
        ------
        Exception
            If verification fails
        """
        raise NotImplementedError()


class _MockFuture:

    __slots__ = ("val",)

    def __init__(self, val):
        self.val = val

    def result(self):
        return self.val


class SerialClient(AbstractClient):
    def submit(self, task_fn, *args, **kwargs):
        return _MockFuture(task_fn(*args, **kwargs))

    def verify(self):
        return


class ProcessPoolClient(AbstractClient):
    def __init__(self, max_workers):
        """
        Generic wrapper around ProcessPoolExecutor. Each call to submit()
        will be run on a different worker.  If the number of jobs submitted
        is larger than the number of workers, the jobs will be batched. Each
        worker will run at most one job.

        Parameters
        ----------
        max_workers: int
            Number of workers to launch via the ProcessPoolExecutor

        """
        ctxt = multiprocessing.get_context("spawn")
        # (ytz): on python <= 3.6 this will throw an exception since mp_context is
        # not supported
        self.executor = futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctxt)
        self.max_workers = max_workers
        self._idx = 0

    def submit(self, task_fn, *args, **kwargs):
        """
        See abstract class for documentation.
        """
        future = self.executor.submit(task_fn, *args)
        self._idx = (self._idx + 1) % self.max_workers
        return future

    def verify(self):
        """
        See abstract class for documentation.
        """
        return


class CUDAPoolClient(ProcessPoolClient):
    """
    Specialized wrapper for CUDA-dependent processes. Each call to submit()
    will run on a different GPU modulo num workers, which should be set to
    the number of GPUs.
    """

    def __init__(self, max_workers):
        super().__init__(max_workers)

    @staticmethod
    def wrapper(max_workers, idx, fn, *args):
        # for a single worker, do not overwrite CUDA_VISIBLE_DEVICES
        # so that multiple single gpu jobs can be run on the same node
        if max_workers > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        return fn(*args)

    def submit(self, task_fn, *args, **kwargs):
        """
        See abstract class for documentation.
        """
        future = self.executor.submit(self.wrapper, self.max_workers, self._idx, task_fn, *args)
        self._idx = (self._idx + 1) % self.max_workers
        return future

    def verify(self):
        """
        See abstract class for documentation.
        """
        gpus = get_gpu_count()
        assert self.max_workers <= gpus, f"More workers '{self.max_workers}' requested than GPUs '{gpus}'"


class BinaryFutureWrapper:
    def __init__(self, future):
        """
        Utility class to help unwrap pickle'd Future objects.
        """
        self._future = future

    def result(self):
        return pickle.loads(self._future.result().binary)


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
        self._idx = (self._idx + 1) % len(self.stubs)
        return BinaryFutureWrapper(future)

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


class AbstractFileClient:
    def store(self, path: str, data: bytes):
        """
        Store the results to the given path.

        Parameters
        ----------
        path:
            Relative path to store the data. The client may interpret
            this path as appropriate (i.e. file path, s3 path, etc).

        data:
            Binary contents to store.
        """
        raise NotImplementedError()

    def load(self, path: str) -> bytes:
        """
        Load the results from the given path.

        Parameters
        ----------
        path:
            Path to load from, the value returned by the `store` method.

        Returns
        -------
        bytes
            Binary contents of the file.
        """
        raise NotImplementedError()

    def exists(self, path: str) -> bool:
        """
        Parameters
        ----------
        path:
            Path to load from, the value returned by the `store` method.

        Returns
        -------
        bool
            True if the results exist at this path.
        """
        raise NotImplementedError()

    def full_path(self, path: str) -> str:
        """
        Parameters
        ----------
        path:
            Relative path to use.

        Returns
        -------
        str:
            The full path, the meaning of which depends on the
            subclass.
        """
        raise NotImplementedError()


class FileClient(AbstractFileClient):
    def __init__(self, base: Path = None):
        self.base = base or Path().cwd()

    def store(self, path: str, data: bytes):
        full_path = Path(self.full_path(path))
        full_path.write_bytes(data)

    def load(self, path: str) -> bytes:
        full_path = Path(self.full_path(path))
        return full_path.read_bytes()

    def exists(self, path: str) -> bool:
        return Path(self.full_path(path)).exists()

    def full_path(self, path: str) -> str:
        return str(Path(self.base, path).absolute())


def save_results(result_paths: List[str], local_file_client: FileClient, remote_file_client: AbstractFileClient):
    """
    Load the results from `remote_file_client`, which may be remote and
    store them on the local file system using `local_file_client`.
    """
    for result_path in result_paths:
        if not local_file_client.exists(result_path):
            local_file_client.store(result_path, remote_file_client.load(result_path))
