import gc
import tempfile
import weakref
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, example, given, seed
from hypothesis.extra.numpy import array_shapes, arrays, floating_dtypes, from_dtype
from hypothesis.strategies import composite, integers, lists

from timemachine.fe import stored_arrays
from timemachine.fe.stored_arrays import StoredArrays
from timemachine.parallel.client import FileClient


@composite
def chunks(draw):
    shape = draw(array_shapes())
    dtype = draw(floating_dtypes())
    return draw(lists(arrays(dtype, shape, elements=from_dtype(dtype, allow_subnormal=False)), max_size=3))


def stored_arrays_from_chunks(chunks):
    sa = StoredArrays()
    for chunk in chunks:
        sa.extend(chunk)
    return sa


@given(lists(chunks()))
@seed(2023)
def test_stored_arrays_extend_iter(chunks):
    sa = stored_arrays_from_chunks(chunks)
    for actual, expected in zip(sa, (arr for chunk in chunks for arr in chunk)):
        np.testing.assert_array_equal(actual, expected)


@composite
def lists_of_chunks_with_index(draw):
    shape = draw(array_shapes())
    dtype = draw(floating_dtypes())
    chunks = lists(min_size=1, elements=arrays(dtype, shape, elements=from_dtype(dtype, allow_subnormal=False)))
    list_of_chunks = draw(lists(min_size=1, elements=chunks))
    n = sum(len(c) for c in list_of_chunks)
    ix = draw(integers(-n, n - 1))
    return list_of_chunks, ix


@given(lists_of_chunks_with_index())
@seed(2023)
def test_stored_arrays_getitem(chunks_index):
    chunks, ix = chunks_index
    sa = stored_arrays_from_chunks(chunks)
    ref = [arr for chunk in chunks for arr in chunk]
    np.testing.assert_array_equal(sa[ix], ref[ix])


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


@given(lists(chunks()))
@seed(2023)
def test_stored_arrays_eq(chunks):
    a = StoredArrays()
    b = StoredArrays()
    for chunk in chunks:
        a.extend(chunk)
        b.extend(chunk)

    assert a is not b
    assert a == b


@given(lists(chunks(), max_size=3), lists(chunks(), max_size=3))
@seed(2023)
@example([np.array([]), np.array([1])], [np.array([1])])
def test_stored_arrays_neq(chunks1, chunks2):
    assume(not all(np.array_equal(a, b, equal_nan=True) for a, b in zip(chunks1, chunks2)))
    a = StoredArrays()
    for chunk in chunks1:
        a.extend(chunk)
    b = StoredArrays()
    for chunk in chunks2:
        b.extend(chunk)
    assert a != b


def test_stored_arrays_cleanup():
    sa = StoredArrays()
    sa.extend([np.array([1, 2, 3])])
    path = sa._dir
    assert path.exists()

    del sa
    gc.collect()
    assert not path.exists()


stored_arrays_instances = lists(chunks()).map(stored_arrays_from_chunks)


@contextmanager
def file_client():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield FileClient(Path(temp_dir))


@given(stored_arrays_instances)
@seed(2023)
def test_stored_arrays_store_load_roundtrip(sa_ref):
    with file_client() as fc:
        stored_arrays.store(sa_ref, fc)
        sa_test = stored_arrays.load(fc)
    assert sa_ref == sa_test


def test_stored_arrays_store_raises_on_file_collision():
    with file_client() as fc:
        sa = stored_arrays_from_chunks([np.array([1, 2, 3])])
        stored_arrays.store(sa, fc)

        with pytest.raises(FileExistsError):
            stored_arrays.store(sa, fc)

        stored_arrays.store(sa, fc, prefix=Path("subdir"))  # no collision
