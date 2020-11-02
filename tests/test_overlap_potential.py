import functools
import numpy as np

from timemachine.potentials import shape

def test_overlap():

    A = np.random.rand(5, 3)*10
    B = np.random.rand(6, 3)*10

    # A = np.random.rand(1, 3)*10
    # B = A

    N = A.shape[0] + B.shape[0]

    a_idxs = np.arange(A.shape[0])
    b_idxs = np.arange(B.shape[0]) + A.shape[0]

    x = np.concatenate([A, B])

    # params = np.array([[0.2, 1.0], [0.5, 1.0]])

    params = np.random.rand(N, 2)/10

    u_fn = functools.partial(shape.overlap,
        params=params,
        a_idxs=a_idxs,
        b_idxs=b_idxs)

    volumes = u_fn(x)


    assert np.all(volumes <= 1)
    assert np.all(volumes >= 0)
