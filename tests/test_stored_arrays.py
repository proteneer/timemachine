import gc
import pickle
import tempfile
import weakref
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, example, given, seed
from hypothesis.extra.numpy import array_shapes, arrays, floating_dtypes, from_dtype
from hypothesis.strategies import composite, integers, lists, none, one_of

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
def test_stored_arrays_getitem_index(chunks_index):
    chunks, ix = chunks_index
    sa = stored_arrays_from_chunks(chunks)
    ref = [arr for chunk in chunks for arr in chunk]
    np.testing.assert_array_equal(sa[ix], ref[ix])


@composite
def lists_of_chunks_with_slice(draw):
    shape = draw(array_shapes())
    dtype = draw(floating_dtypes())
    chunks = lists(min_size=1, elements=arrays(dtype, shape, elements=from_dtype(dtype, allow_subnormal=False)))
    list_of_chunks = draw(lists(min_size=1, elements=chunks))
    n = sum(len(c) for c in list_of_chunks)
    idx = integers(-n, n - 1)
    start = draw(one_of(idx, none()))
    stop = draw(one_of(idx, none()))
    step = draw(one_of(integers().filter(lambda n: n != 0), none()))
    return list_of_chunks, slice(start, stop, step)


@given(lists_of_chunks_with_slice())
@example(([np.array([1, 2, 3]), np.array([4, 5, 6])], slice(None)))
@example(([np.array([1, 2, 3]), np.array([4, 5, 6])], slice(1)))
@example(([np.array([1, 2, 3]), np.array([4, 5, 6])], slice(2, 4)))
@example(([np.array([1, 2, 3]), np.array([4, 5, 6])], slice(None, None, 2)))
@seed(2023)
def test_stored_arrays_getitem_slice(chunks_slice):
    chunks, slc = chunks_slice
    sa = stored_arrays_from_chunks(chunks)
    ref = [arr for chunk in chunks for arr in chunk]
    np.testing.assert_array_equal(sa[slc], ref[slc])


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
    assert ref() is not None

    del data
    gc.collect()

    # data should have been GC'd
    assert ref() is None


@given(lists(chunks()))
@seed(2023)
def test_stored_arrays_eq(chunks):
    a = stored_arrays_from_chunks(chunks)
    b = stored_arrays_from_chunks(chunks)
    assert a == a
    assert a is not b
    assert a == b


@given(lists(chunks(), max_size=3), lists(chunks(), max_size=3))
@seed(2023)
@example([np.array([]), np.array([1])], [np.array([1])])
def test_stored_arrays_neq(chunks1, chunks2):
    assume(not all(np.array_equal(a, b, equal_nan=True) for a, b in zip(chunks1, chunks2)))
    a = stored_arrays_from_chunks(chunks1)
    b = stored_arrays_from_chunks(chunks2)
    assert a != b


def test_stored_arrays_cleanup():
    sa = StoredArrays()
    sa.extend([np.array([1, 2, 3])])
    path = sa._path()
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
        sa_ref.store(fc)
        sa_test = StoredArrays.load(fc)
    assert sa_ref == sa_test


def test_stored_arrays_store_raises_on_file_collision():
    with file_client() as fc:
        sa = stored_arrays_from_chunks([np.array([1, 2, 3])])
        sa.store(fc)

        with pytest.raises(FileExistsError):
            sa.store(fc)

        sa.store(fc, prefix=Path("subdir"))  # no collision


def test_stored_arrays_raises_on_pickling_attempt():
    sa = StoredArrays()
    with pytest.raises(NotImplementedError) as e:
        _ = pickle.dumps(sa)
    assert "pickling not implemented" in str(e)
