import numpy as np
import time

import multiprocessing

import pickle
import grpc

from parallel import service_pb2_grpc, service_pb2

from concurrent import futures

import os

# (ytz): The classes in this file are designed to help provide a consistent API between
# multiprocessing (typically for local cluster use) and gRPC (distributed and multi-node).

class AbstractClient():

    def submit(self, task_fn, *args):
        """
        Submit is an asynchronous method that will launch task_fn whose 
        results will be collected a later point in time. The input task_fn
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
        if multiprocessing.get_start_method() != 'spawn':
            message = "Jax and CUDA are not fork-safe! Please call `multiprocessing.set_start_method('spawn')` and try again"
            raise (RuntimeError(message))

        self.executor = futures.ProcessPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers
        self._idx = 0

    def submit(self, task_fn, *args):
        """
        See abstract class for documentation.
        """
        future = self.executor.submit(task_fn, *args)
        self._idx = (self._idx + 1) % self.max_workers
        return future

class CUDAPoolClient(ProcessPoolClient):
    """
    Specialized wrapper for CUDA-dependent processes. Each call to submit()
    will run on a different GPU modulo num workers, which should be set to
    the number of GPUs.
    """

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

class BinaryFutureWrapper():
    """
    Utility class to help unwrap pickle'd Future objects.
    """
    def __init__(self, future):
        self._future = future

    def result(self):
        return pickle.loads(self._future.result().binary)

class GRPCClient(AbstractClient):

    def __init__(self, stubs):
        """
        GRPCClient is meant for distributed use. The GRPC workers must
        be launched prior to starting the client. The worker version should be compatible
        with that of client, as pickling requires that the files are consistent
        across the client and the server.

        Parameters
        ----------
        stubs: list
            Initialized grpc stubs to be used

        """

        self.stubs = stubs
        self._idx = 0

    def submit(self, task_fn, args):
        """
        See abstract class for documentation.
        """
        binary = pickle.dumps((task_fn, args))
        request = service_pb2.PickleData(binary=binary)
        future = self.stubs[self._idx].Submit.future(request)
        self._idx = (self._idx + 1) % len(self.stubs)
        return BinaryFutureWrapper(future)
