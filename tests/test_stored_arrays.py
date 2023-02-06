import gc
import weakref

import numpy as np
import pytest
from hypothesis import given, seed
from hypothesis.extra.numpy import array_shapes, arrays, floating_dtypes
from hypothesis.strategies import composite, integers, lists

from timemachine.fe.stored_arrays import StoredArrays


@composite
def chunks(draw):
    shape = draw(array_shapes())
    dtype = draw(floating_dtypes())
    return draw(lists(arrays(dtype, shape)))


@given(lists(chunks()))
@seed(2023)
def test_stored_arrays_extend_iter_roundtrip(chunks):
    sas = StoredArrays()
    for chunk in chunks:
        sas.extend(chunk)

    for actual, expected in zip(sas, (arr for chunk in chunks for arr in chunk)):
        np.testing.assert_array_equal(actual, expected)


@composite
def lists_of_chunks_with_index(draw):
    shape = draw(array_shapes())
    dtype = draw(floating_dtypes())
    chunks = lists(min_size=1, elements=arrays(dtype, shape))
    list_of_chunks = draw(lists(min_size=1, elements=chunks))
    n = sum(len(c) for c in list_of_chunks)
    ix = draw(integers(-n, n - 1))
    return list_of_chunks, ix


@given(lists_of_chunks_with_index())
@seed(2023)
def test_stored_arrays_getitem(chunks_index):
    chunks, ix = chunks_index
    sas = StoredArrays()
    for chunk in chunks:
        sas.extend(chunk)

    ref = [arr for chunk in chunks for arr in chunk]
    np.testing.assert_array_equal(sas[ix], ref[ix])


@pytest.mark.parametrize(
    "ctor",
    [
        pytest.param(list, marks=pytest.mark.xfail(reason="expect list to hold refs", strict=True)),
        pytest.param(StoredArrays),
    ],
)
def test_stored_arrays_doesnt_hold_references(ctor):
    xs = ctor()
    data = np.arange(10)
    xs.extend([data])

    ref = weakref.ref(data)
    del data
    gc.collect()

    # data should have been GC'd
    assert ref() is None
