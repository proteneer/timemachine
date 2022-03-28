import pytest

pytestmark = [pytest.mark.nogpu]


def test_import_lib_potentials_without_custom_ops():
    import timemachine.lib.potentials  # noqa: F401
