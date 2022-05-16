# tests for parallel execution
import concurrent
import os
import random
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

import grpc
from timemachine import parallel
from timemachine.parallel import client, worker
from timemachine.parallel.utils import batch_list, flatten_list


def jax_fn(x):
    return jnp.sqrt(x)


def square(a):
    return a * a


def mult(x, y):
    return x * y


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

        np.testing.assert_array_equal(test_res, arr * arr)

    def test_jax(self):
        # (ytz): test that jax code can be launched via multiprocessing
        # if we didn't set get_context('spawn') earlier then this will hang.
        x = jnp.array([50.0, 2.0])
        fut = self.cli.submit(jax_fn, x)
        np.testing.assert_almost_equal(fut.result(), np.sqrt(x))


def environ_check():
    return os.environ["CUDA_VISIBLE_DEVICES"]


class TestGPUCount(unittest.TestCase):
    @patch("timemachine.parallel.utils.check_output")
    def test_get_gpu_count(self, mock_output):
        mock_output.return_value = b"\n".join([f"GPU #{i}".encode() for i in range(5)])
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""}):
            assert parallel.utils.get_gpu_count() == 5

        mock_output.return_value = b"\n".join([f"GPU #{i}".encode() for i in range(100)])
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""}):
            assert parallel.utils.get_gpu_count() == 100

        # Respect CUDA_VISIBLE_DEVICES if present
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,5,10"}):
            assert parallel.utils.get_gpu_count() == 3

        mock_output.side_effect = FileNotFoundError("nvidia-smi missing")
        assert parallel.utils.get_gpu_count() == 0


class TestCUDAPoolClient(unittest.TestCase):
    def setUp(self):
        self.max_workers = 2
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

        np.testing.assert_array_equal(test_res, expected)

    def test_too_many_workers(self):
        # I look forward to the day that we have 814 GPUs
        cli = client.CUDAPoolClient(814)
        with self.assertRaises(AssertionError):
            cli.verify()

    def test_single_worker(self):
        cli = client.CUDAPoolClient(1)
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "123"}):
            result = cli.submit(environ_check).result()
        assert result == "123"


class TestGRPCClient(unittest.TestCase):
    def setUp(self):

        starting_port = random.randint(2000, 5000)
        # setup server, in reality max_workers is equal to number of gpus
        self.ports = [starting_port + i for i in range(2)]
        self.servers = []
        for port in self.ports:
            server = grpc.server(
                concurrent.futures.ThreadPoolExecutor(max_workers=1),
                options=[
                    ("grpc.max_send_message_length", 50 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 50 * 1024 * 1024),
                ],
            )
            parallel.grpc.service_pb2_grpc.add_WorkerServicer_to_server(worker.Worker(), server)
            server.add_insecure_port("[::]:" + str(port))
            server.start()
            self.servers.append(server)

        # setup client
        self.hosts = [f"0.0.0.0:{port}" for port in self.ports]
        self.cli = client.GRPCClient(self.hosts)

    @patch("timemachine.parallel.worker.get_worker_status")
    def test_checking_host_status(self, mock_status):
        # All the workers return the same thing
        mock_status.side_effect = [
            parallel.grpc.service_pb2.StatusResponse(nvidia_driver="foo", git_sha="bar") for _ in self.servers
        ]
        self.cli.verify()

        mock_status.side_effect = [
            parallel.grpc.service_pb2.StatusResponse(nvidia_driver=f"foo{i}", git_sha=f"bar{i}")
            for i in range(len(self.servers))
        ]

        with self.assertRaises(AssertionError):
            self.cli.verify()

    @patch("timemachine.parallel.worker.get_worker_status")
    def test_unavailable_host(self, mock_status):

        hosts = self.hosts.copy()
        bad_host = "128.128.128.128:8888"
        hosts.append(bad_host)  # Give it a bad connexion, should fail
        cli = client.GRPCClient(hosts)

        # All the workers return the same thing
        mock_status.side_effect = [
            parallel.grpc.service_pb2.StatusResponse(nvidia_driver="foo", git_sha="bar") for _ in self.servers
        ]

        with self.assertRaises(AssertionError) as e:
            cli.verify()
        self.assertIn(bad_host, str(e.exception))

    def test_default_port(self):
        host = "128.128.128.128"
        cli = client.GRPCClient([host], default_port=9999)
        self.assertEqual(cli.hosts[0], "128.128.128.128:9999")

    def test_hosts_from_file(self):
        with self.assertRaises(AssertionError):
            cli = client.GRPCClient("nosuchfile", default_port=9999)

        hosts = ["128.128.128.128", "127.127.127.127:8888"]
        with NamedTemporaryFile(suffix=".txt") as temp:
            for host in hosts:
                temp.write(f"{host}\n".encode("utf-8"))
            temp.flush()
            cli = client.GRPCClient(temp.name, default_port=9999)
            self.assertEqual(cli.hosts[0], "128.128.128.128:9999")
            self.assertEqual(cli.hosts[1], hosts[1])

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

        np.testing.assert_array_equal(test_res, xs * ys)

    def test_foo_1_arg(self):
        xs = np.linspace(0, 1.0, 5)

        futures = []
        for x in xs:
            fut = self.cli.submit(square, x)
            futures.append(fut)

        test_res = []
        for f in futures:
            test_res.append(f.result())

        np.testing.assert_array_equal(test_res, xs * xs)

    def tearDown(self):
        for server in self.servers:
            server.stop(5)


def test_batch_list():
    assert batch_list(list(range(10)), 5) == [[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]]


def test_flatten_list():
    assert flatten_list([[0, 1, 2], [3, 4], [5, 6]]) == list(range(7))


def test_file_client(tmpdir):
    with tmpdir.as_cwd():
        tmpdir.mkdir("subdir")
        fc = client.FileClient("subdir")
        fc.store("test", b"data")
        assert fc.exists("test")
        assert str(fc.full_path("test")) == str(Path(tmpdir, "subdir", "test"))
        assert fc.load("test") == b"data"


def test_save_results(tmpdir):
    with tmpdir.as_cwd():
        tmpdir.mkdir("remote")
        rfc = client.FileClient("remote")
        rfc.store("test", b"data")
        rfc.store("test2", b"data")

        tmpdir.mkdir("local")
        lfc = client.FileClient("local")

        result_paths = ["test", "test2"]
        client.save_results(result_paths, lfc, rfc)

        for result_path in result_paths:
            assert lfc.exists(result_path)
            assert Path("local", result_path).exists()
