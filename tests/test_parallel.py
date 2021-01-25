# tests for parallel execution
import numpy as np

import parallel
from parallel import client
from parallel import worker

import unittest
import os

import grpc
import concurrent

import jax.numpy as jnp


def jax_fn(x):
    return jnp.sqrt(x)

def square(a):
    return a*a

class TestProcessPool(unittest.TestCase):

    def setUp(self):
        max_workers = 10
        self.cli = client.ProcessPoolClient(max_workers)

    def test_submit(self):

        arr = np.linspace(0, 1.0, 5)

        futures = []
        for x in arr:
            fut = self.cli.submit(square, x)
            futures.append(fut)

        test_res = []
        for f in futures:
            test_res.append(f.result())

        np.testing.assert_array_equal(test_res, arr*arr)

    def test_jax(self):
        # (ytz): test that jax code can be launched via multiprocessing
        # if we didn't set get_context('spawn') earlier then this will hang.
        x = jnp.array([50., 2.0])
        fut = self.cli.submit(jax_fn, x)
        np.testing.assert_almost_equal(fut.result(), np.sqrt(x))

def environ_check():
    return os.environ['CUDA_VISIBLE_DEVICES']

class TestCUDAPoolClient(unittest.TestCase):

    def setUp(self):
        max_workers = 4
        self.cli = client.CUDAPoolClient(max_workers)

    def test_submit(self):

        futures = []
        for _ in range(10):
            fut = self.cli.submit(environ_check)
            futures.append(fut)

        test_res = []
        for f in futures:
            test_res.append(f.result())

        np.testing.assert_array_equal(
            test_res,
            ['0', '1', '2', '3', '0', '1', '2', '3', '0', '1']
        )

class TestGRPCClient(unittest.TestCase):

    def setUp(self):

        # setup server, in reality max_workers is equal to number of gpus
        ports = [2020, 2021]
        self.servers = []
        for port in ports:
            server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=2),
                options = [
                    ('grpc.max_send_message_length', 50 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 50 * 1024 * 1024)
                ]
            )
            parallel.service_pb2_grpc.add_WorkerServicer_to_server(worker.Worker(), server)
            server.add_insecure_port('[::]:'+str(port))
            server.start()
            self.servers.append(server)

        # setup client
        stubs = []
        for port in ports:
            stubs = []
            channel = grpc.insecure_channel('0.0.0.0:'+str(port),
                options = [
                    ('grpc.max_send_message_length', 500 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 500 * 1024 * 1024)
                ]
            )
            stub = parallel.service_pb2_grpc.WorkerStub(channel)
            stubs.append(stub)

        self.cli = client.GRPCClient(stubs)

    def test_foo(self):
        arr = np.linspace(0, 1.0, 5)

        futures = []
        for x in arr:
            fut = self.cli.submit(square, x)
            futures.append(fut)

        test_res = []
        for f in futures:
            test_res.append(f.result())

        np.testing.assert_array_equal(test_res, arr*arr)

    def tearDown(self):
        for server in self.servers:
            server.stop(5)
