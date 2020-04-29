from hypothesis import given

from locus.core.utils import ceil_division
from . import strategies


@given(strategies.integers, strategies.non_zero_integers)
def test_basic(dividend: int, divisor: int) -> None:
    result = ceil_division(dividend, divisor)

    assert isinstance(result, int)


@given(strategies.integers, strategies.non_zero_integers)
def test_properties(dividend: int, divisor: int) -> None:
    result = ceil_division(dividend, divisor)

    quotient, remainder = divmod(dividend, divisor)

    assert result == quotient + bool(remainder)
