# tests for parallel execution
import numpy as np

import parallel
from parallel import client
from parallel import worker
from parallel.utils import get_gpu_count

import unittest
from unittest.mock import patch
import os

import grpc
import concurrent

import jax.numpy as jnp


def jax_fn(x):
    return jnp.sqrt(x)

def square(a):
    return a*a

def mult(x,y):
    return x*y

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

class TestGPUCount(unittest.TestCase):

    @patch("parallel.utils.check_output")
    def test_get_gpu_count(self, mock_output):
        mock_output.return_value = b"\n".join([f"GPU #{i}".encode() for i in range(5)])
        assert parallel.utils.get_gpu_count() == 5

        mock_output.return_value = b"\n".join([f"GPU #{i}".encode() for i in range(100)])
        assert parallel.utils.get_gpu_count() == 100

        mock_output.side_effect = FileNotFoundError("nvidia-smi missing")
        with self.assertRaises(FileNotFoundError):
            parallel.utils.get_gpu_count()

class TestCUDAPoolClient(unittest.TestCase):

    def setUp(self):
        self.max_workers = get_gpu_count()
        self.cli = client.CUDAPoolClient(self.max_workers)

    def test_submit(self):

        operations = 10

        futures = []
        for _ in range(operations):
            fut = self.cli.submit(environ_check)
            futures.append(fut)

        test_res = []
        for f in futures:
            test_res.append(f.result())

        expected = [str(i % self.max_workers) for i in range(operations)]

        np.testing.assert_array_equal(
            test_res,
            expected
        )

    def test_too_many_workers(self):
        # I look forward to the day that we have 814 GPUs
        cli = client.CUDAPoolClient(814)
        with self.assertRaises(AssertionError):
            cli.verify()

class TestGRPCClient(unittest.TestCase):

    def setUp(self):

        # setup server, in reality max_workers is equal to number of gpus
        ports = [2020 + i for i in range(4)]
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
        hosts = [f"0.0.0.0:{port}" for port in ports]
        self.cli = client.GRPCClient(hosts)

    def test_foo_2_args(self):
        xs = np.linspace(0, 1.0, 5)
        ys = np.linspace(1.2, 2.2, 5)

        futures = []
        for x, y in zip(xs, ys):
            fut = self.cli.submit(mult, x, y)
            futures.append(fut)

        test_res = []
        for f in futures:
            test_res.append(f.result())

        np.testing.assert_array_equal(test_res, xs*ys)

    def test_foo_1_arg(self):
        xs = np.linspace(0, 1.0, 5)

        futures = []
        for x in xs:
            fut = self.cli.submit(square, x)
            futures.append(fut)

        test_res = []
        for f in futures:
            test_res.append(f.result())

        np.testing.assert_array_equal(test_res, xs*xs)

    def tearDown(self):
        for server in self.servers:
            server.stop(5)
