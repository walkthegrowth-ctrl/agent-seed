from agent_seed.tools import ToolError, calculator


def test_basic_arithmetic():
    assert calculator("2 + 2") == 4.0
    assert calculator("10 - 4") == 6.0
    assert calculator("3 * 4") == 12.0
    assert calculator("7 / 2") == 3.5


def test_operators():
    assert calculator("2 ** 10") == 1024.0
    assert calculator("17 // 5") == 3.0
    assert calculator("44 % 7") == 2.0


def test_unary_and_parens():
    assert calculator("-5 + 3") == -2.0
    assert calculator("-(2 + 3)") == -5.0
    assert calculator("((1 + 2) * (3 + 4))") == 21.0


def test_floats():
    assert calculator("1.5 * 2") == 3.0


def test_rejects_non_numeric_constants():
    for bad in ("'abc'", "(1, 2)", "[1] + 1", "x + 1"):
        try:
            calculator(bad)
        except ToolError:
            pass
        else:
            raise AssertionError(f"expected ToolError for {bad!r}")


def test_division_by_zero():
    try:
        calculator("1 / 0")
    except ToolError:
        return
    raise AssertionError("expected ToolError for division by zero")


def test_invalid_expression():
    for bad in ("not an expression", "1 +", "2**", ""):
        try:
            calculator(bad)
            raise AssertionError(f"expected ToolError for {bad!r}")
        except ToolError:
            pass
