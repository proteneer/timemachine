from typing import List, Optional, Any
import numpy as np
import time

import multiprocessing

import pickle
import grpc

from parallel import service_pb2_grpc, service_pb2
from parallel.utils import get_gpu_count

from concurrent import futures

import sys
import os

# (ytz): The classes in this file are designed to help provide a consistent API between
# multiprocessing (typically for local cluster use) and gRPC (distributed and multi-node).

class AbstractClient():

    def submit(self, task_fn, *args):
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
        ctxt = multiprocessing.get_context('spawn')
        # (ytz): on python <= 3.6 this will throw an exception since mp_context is
        # not supported
        self.executor = futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctxt)
        self.max_workers = max_workers
        self._idx = 0

    def submit(self, task_fn, *args):
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
    def wrapper(idx, fn, *args):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(idx)
        return fn(*args)

    def submit(self, task_fn, *args):
        """
        See abstract class for documentation.
        """
        future = self.executor.submit(self.wrapper, self._idx, task_fn, *args)
        self._idx = (self._idx + 1) % self.max_workers
        return future

    def verify(self):
        """
        See abstract class for documentation.
        """
        gpus = get_gpu_count()
        assert self.max_workers <= gpus, f"More workers '{self.max_workers}' requested than GPUs '{gpus}'"

class BinaryFutureWrapper():

    def __init__(self, future):
        """
        Utility class to help unwrap pickle'd Future objects.
        """
        self._future = future

    def result(self):
        return pickle.loads(self._future.result().binary)

class GRPCClient(AbstractClient):

    def __init__(self, hosts: List[str], options: Optional[List[Any]] = None):
        """
        GRPCClient is meant for distributed use. The GRPC workers must
        be launched prior to starting the client. The worker version should be compatible
        with that of client, as pickling requires that the files are consistent
        across the client and the server.

        Parameters
        ----------
        hosts: list
            List of hosts to use as GRPC workers.
        options: list
            List of options to configure GRPC connections

        """
        self.hosts = hosts
        self.stubs = []
        if options is None:
            options = [
                ('grpc.max_send_message_length', 500 * 1024 * 1024),
                ('grpc.max_receive_message_length', 500 * 1024 * 1024)
            ]
        for host in hosts:
            channel = grpc.insecure_channel(
                host,
                options=options,
            )
            self.stubs.append(service_pb2_grpc.WorkerStub(channel))
        self._idx = 0

    def submit(self, task_fn, *args):
        """
        See abstract class for documentation.
        """
        binary = pickle.dumps((task_fn, args))
        request = service_pb2.PickleData(binary=binary)
        future = self.stubs[self._idx].Submit.future(request)
        self._idx = (self._idx + 1) % len(self.stubs)
        return BinaryFutureWrapper(future)

    def verify(self):
        """
        See abstract class for documentation.
        """
        futures = [stub.Status.future(service_pb2.StatusRequest()) for stub in self.stubs]
        workers_status = [x.result() for x in futures]
        for field in ("nvidia_driver", "git_sha",):
            # All fields should be the same
            host_values = {self.hosts[i]: getattr(x, field) for i, x in enumerate(workers_status)}
            uni_vals = set(host_values.values())
            assert len(uni_vals) == 1, f"Not all hosts agreed for {field}: {host_values}"
            assert all(uni_vals), f"Missing values for {field}: {host_values}"
