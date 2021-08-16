import io
from contextlib import redirect_stdout
from fe.free_energy_rabfe import RABFEResult


def capture_print(closure):
    """capture anything printed when we call closure()"""

    f = io.StringIO()
    with redirect_stdout(f):
        closure()
    printed = f.getvalue()
    return printed


def test_rabfe_result_to_from_log():
    """assert equality after round-trip to/from preferred terminal log format"""

    result = RABFEResult('my mol', 1.0, 2.0, 3.0, 4.0)

    printed = capture_print(result.log)
    first_line = printed.splitlines()[0]

    reconstructed = RABFEResult.from_log(first_line)
    assert result == reconstructed
