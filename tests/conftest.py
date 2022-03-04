# This file defines fixtures shared across multiple test modules.
# https://docs.pytest.org/en/latest/how-to/fixtures.html#scope-sharing-fixtures-across-classes-modules-packages-or-session

import gc

import pytest

from timemachine.lib import custom_ops


@pytest.fixture(autouse=True)
def reset_cuda_device_after_test(request):
    """Calls cudaDeviceReset() after each test.

    This is needed for 'cuda-memcheck --leak-check full' to catch leaks"""

    yield

    # If the test is not marked for memory tests, no need to reset device
    if "memcheck" not in request.keywords:
        return
    # ensure that destructors are called before cudaDeviceReset()
    gc.collect()

    custom_ops.cuda_device_reset()
