import numpy as np
from fe.loss import l1_loss, pseudo_huber_loss, flat_bottom_loss


def _assert_nonnegative_loss(loss_fxn: callable):
    residuals = np.linspace(-10, 10, 100)
    losses = np.array([loss_fxn(res) for res in residuals])
    np.testing.assert_array_less(0, losses)


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
