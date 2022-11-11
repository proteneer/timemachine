import multiprocessing
import os
import pickle
from concurrent import futures
from pathlib import Path
from typing import List

from timemachine.parallel.utils import get_gpu_count

# (ytz): The classes in this file are designed to help provide a consistent API between
# multiprocessing (typically for local cluster use) and gRPC (distributed and multi-node).


class AbstractClient:
    def __init__(self):
        self.max_workers = 1

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

    @property
    def id(self) -> str:
        """
        Return the id as a str for this subjob
        """
        return "1"

    @property
    def name(self) -> str:
        """
        Return the name as a str for this subjob
        """
        return "1"


class WrappedFuture:
    def __init__(self, future, job_id: str):
        self._future = future
        self._id = job_id

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return str(self._id)

    def __getattr__(self, attr):
        return getattr(self._future, attr)


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
        self.max_workers = max_workers
        self._idx = 0
        self._total_idx = 0
        ctxt = multiprocessing.get_context("spawn")
        self.executor = futures.ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctxt)

    def submit(self, task_fn, *args, **kwargs):
        """
        See abstract class for documentation.
        """
        future = self.executor.submit(task_fn, *args, **kwargs)
        job_id = str(self._total_idx)
        self._total_idx += 1
        self._idx = (self._idx + 1) % self.max_workers
        return WrappedFuture(future, job_id)

    def verify(self):
        """
        See abstract class for documentation.
        """
        return

    def __getstate__(self):
        # Only store the max workers in the pickle
        return (self.max_workers,)

    def __setstate__(self, state):
        max_workers = state[0]
        self.__init__(max_workers)


class CUDAPoolClient(ProcessPoolClient):
    """
    Specialized wrapper for CUDA-dependent processes. Each call to submit()
    will run on a different GPU modulo num workers, which should be set to
    the number of GPUs.
    """

    def __init__(self, max_workers):
        super().__init__(max_workers)

    @staticmethod
    def wrapper(max_workers, idx, fn, *args, **kwargs):
        # for a single worker, do not overwrite CUDA_VISIBLE_DEVICES
        # so that multiple single gpu jobs can be run on the same node
        if max_workers > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        return fn(*args, **kwargs)

    def submit(self, task_fn, *args, **kwargs):
        """
        See abstract class for documentation.
        """
        future = self.executor.submit(self.wrapper, self.max_workers, self._idx, task_fn, *args, **kwargs)
        job_id = str(self._total_idx)
        self._total_idx += 1
        self._idx = (self._idx + 1) % self.max_workers
        return WrappedFuture(future, job_id)

    def verify(self):
        """
        See abstract class for documentation.
        """
        gpus = get_gpu_count()
        assert self.max_workers <= gpus, f"More workers '{self.max_workers}' requested than GPUs '{gpus}'"


class BinaryFutureWrapper:
    def __init__(self, future, job_id):
        """
        Utility class to help unwrap pickle'd Future objects.
        """
        self._future = future
        self._id = job_id

    def result(self):
        return pickle.loads(self._future.result().binary)

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return str(self._id)


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
        full_path.parent.mkdir(parents=True, exist_ok=True)
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
