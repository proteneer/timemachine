import pytest

pytestmark = [pytest.mark.nocuda]


def test_import_potentials_without_custom_ops():
    import timemachine.potentials  # noqa: F401
