import numpy as np
from fe.loss import l1_loss, pseudo_huber_loss, flat_bottom_loss, truncated_residuals


def _assert_nonnegative_loss(loss_fxn: callable):
    residuals = np.linspace(-10, 10, 100)
    losses = np.array([loss_fxn(res) for res in residuals])
    np.testing.assert_array_equal(losses >= 0, True)


def _assert_zero_loss_when_perfect(loss_fxn: callable):
    np.testing.assert_almost_equal(loss_fxn(0.0), 0)


def _assert_basic_loss_properties(loss_fxn: callable):
    _assert_nonnegative_loss(loss_fxn)
    _assert_zero_loss_when_perfect(loss_fxn)


def test_l1_loss():
    _assert_basic_loss_properties(l1_loss)


def test_pseudo_huber_loss():
    _assert_basic_loss_properties(pseudo_huber_loss)


def test_flat_bottom_loss():
    _assert_basic_loss_properties(flat_bottom_loss)


def _scalar_truncated_residual(prediction, label, reliable_interval):
    lower, upper = reliable_interval
    assert lower < upper

    if (label >= lower) and (label <= upper):
        residual = prediction - label
    elif label < lower:
        residual = max(0, prediction - lower)
    elif label > upper:
        residual = min(0, prediction - upper)
    else:
        raise (RuntimeError('unsatisfiable reliable_range'))

    return residual


def test_compute_residuals():
    predictions = np.random.rand(10000) * 4 - 2
    labels = np.random.rand(10000) * 4 - 2

    reliable_interval = (-1, +1)

    ref = np.array([
        _scalar_truncated_residual(p, l, reliable_interval)
        for (p, l) in zip(predictions, labels)
    ])

    test = truncated_residuals(predictions, labels, reliable_interval)

    np.testing.assert_allclose(ref, test)
