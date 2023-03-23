# tests for parallel execution
import io
import os
import pickle
import unittest
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from timemachine import parallel
from timemachine.parallel import client
from timemachine.parallel.utils import batch_list

pytestmark = [pytest.mark.nogpu]


def jax_fn(x):
    return jnp.sqrt(x)


def square(a):
    return a * a


def mult(x, y):
    return x * y


def sum(*args, **kwargs):
    return args[0] + kwargs["key"]


class TestProcessPool(unittest.TestCase):
    def setUp(self):
        max_workers = 10
        self.cli = client.ProcessPoolClient(max_workers)

    def test_submit(self):
        assert self.cli.max_workers == 10
        arr = np.linspace(0, 1.0, 5)

        futures = []
        for x in arr:
            fut = self.cli.submit(square, x)
            futures.append(fut)

        test_res = []
        test_ids = []
        test_names = []
        for f in futures:
            test_res.append(f.result())
            test_ids.append(f.id)
            test_names.append(f.name)

        expected_ids = ["0", "1", "2", "3", "4"]
        assert test_ids == expected_ids
        assert test_names == expected_ids

        np.testing.assert_array_equal(test_res, arr * arr)

    def test_submit_kwargs(self):
        arr = np.linspace(0, 1.0, 5)

        futures = []
        for x in arr:
            fut = self.cli.submit(sum, x, key=x)
            futures.append(fut)

        test_res = [f.result() for f in futures]
        np.testing.assert_array_equal(test_res, arr + arr)

    def test_jax(self):
        # (ytz): test that jax code can be launched via multiprocessing
        # if we didn't set get_context('spawn') earlier then this will hang.
        x = jnp.array([50.0, 2.0])
        fut = self.cli.submit(jax_fn, x)
        np.testing.assert_almost_equal(fut.result(), np.sqrt(x))

    def test_pickle(self):
        # test that the client can be pickled
        cli = pickle.loads(pickle.dumps(self.cli))
        assert cli.submit(square, 4).result() == 16


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
        assert self.cli.max_workers == 2

        operations = 10

        futures = []
        for _ in range(operations):
            fut = self.cli.submit(environ_check)
            futures.append(fut)

        test_res = []
        test_ids = []
        test_names = []
        for f in futures:
            test_res.append(f.result())
            test_ids.append(f.id)
            test_names.append(f.name)

        expected_ids = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        assert test_ids == expected_ids
        assert test_names == expected_ids

        expected = [str(i % self.max_workers) for i in range(operations)]

        np.testing.assert_array_equal(test_res, expected)

    def test_submit_kwargs(self):
        arr = np.linspace(0, 1.0, 5)

        futures = []
        for x in arr:
            fut = self.cli.submit(sum, x, key=x)
            futures.append(fut)

        test_res = [f.result() for f in futures]
        np.testing.assert_array_equal(test_res, arr + arr)

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


def test_batch_list():
    assert batch_list(list(range(10)), 5) == [[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]]
    assert batch_list(list(range(10)), None) == [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]


def test_file_client(tmpdir):
    with tmpdir.as_cwd():
        fc = client.FileClient("subdir")
        fc.store("test", b"data")
        assert fc.exists("test")
        assert str(fc.full_path("test")) == str(Path(tmpdir, "subdir", "test"))
        assert fc.load("test") == b"data"

        fc.store_stream("test_copy", io.BytesIO(fc.load("test")))
        assert fc.exists("test_copy")
        assert str(fc.full_path("test_copy")) == str(Path(tmpdir, "subdir", "test_copy"))
        assert fc.load("test") == fc.load("test_copy")

        large_obj = b"a" * (io.DEFAULT_BUFFER_SIZE * 10)
        fc.store_stream("larger_than_stream", io.BytesIO(large_obj))
        assert fc.load("larger_than_stream") == large_obj


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
