# This file defines fixtures shared across multiple test modules.
# https://docs.pytest.org/en/latest/how-to/fixtures.html#scope-sharing-fixtures-across-classes-modules-packages-or-session

import multiprocessing
import os
import warnings

# (ytz): not pretty, but this is needed to get XLA to be less stupid
# see https://github.com/google/jax/issues/1408 for more information
# needs to be set before xla/jax is initialized, and is set to a number
# suitable for running on CI
# NOTE: To have an effect, XLA_FLAGS must be set in the environment before loading JAX (whether directly or transitively
# through another import)
if os.environ.get("XLA_FLAGS"):
    XLA_FLAGS = os.environ.get("XLA_FLAGS")
    warnings.warn(f"Using XLA_FLAGS: {XLA_FLAGS}")
else:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(multiprocessing.cpu_count())

import jax

jax.config.update("jax_enable_x64", True)

import gc

import hypothesis
import pytest

from timemachine.lib import custom_ops

# disable deadlines avoid "Flaky" errors if the first example times out
hypothesis.settings.register_profile("no-deadline", deadline=None)


@pytest.fixture(autouse=True)
def reset_cuda_device_after_test(request):
    """Calls cudaDeviceReset() after each test marked with memcheck.

    This is needed for 'compute-sanitizer --leak-check full' to catch leaks"""

    yield

    # If the test is not marked for memory tests, no need to reset device
    if "memcheck" in request.keywords:
        # ensure that destructors are called before cudaDeviceReset()
        gc.collect()
        custom_ops.cuda_device_reset()
