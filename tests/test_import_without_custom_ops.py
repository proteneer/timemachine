import pytest

pytestmark = [pytest.mark.nogpu]


def test_import_potentials_without_custom_ops():
    import timemachine.potentials  # noqa: F401
