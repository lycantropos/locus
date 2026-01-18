import sys
from decimal import Decimal
from operator import add
from typing import Final

from ground.hints import Box, Point, Segment
from hypothesis import strategies as st

from tests.hints import Scalar, ScalarT
from tests.utils import context, pack, to_pairs

MAX_SCALAR: Final = 10**15
MIN_SCALAR: Final = -MAX_SCALAR


def to_float_strategy(
    min_value: float,
    max_value: float,
    /,
    *,
    allow_nan: bool = False,
    allow_infinity: bool = False,
) -> st.SearchStrategy[float]:
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=allow_nan,
        allow_infinity=allow_infinity,
    ).map(to_digits_count)


def to_digits_count(
    number: float, /, *, max_digits_count: int = sys.float_info.dig
) -> float:
    decimal = Decimal(number).normalize()
    _, significant_digits, exponent = decimal.as_tuple()
    assert isinstance(exponent, int), (number, exponent)
    significant_digits_count = len(significant_digits)
    if exponent < 0:
        fixed_digits_count = (
            1 - exponent
            if exponent <= -significant_digits_count
            else significant_digits_count
        )
    else:
        fixed_digits_count = exponent + significant_digits_count
    if fixed_digits_count <= max_digits_count:
        return number
    whole_digits_count = max(significant_digits_count + exponent, 0)
    if whole_digits_count:
        whole_digits_offset = max(whole_digits_count - max_digits_count, 0)
        decimal /= 10**whole_digits_offset
        whole_digits_count -= whole_digits_offset
    else:
        decimal *= 10 ** (-exponent - significant_digits_count)
        whole_digits_count = 1
    decimal = round(decimal, max(max_digits_count - whole_digits_count, 0))
    return float(str(decimal))


scalar_strategy_strategy: st.SearchStrategy[st.SearchStrategy[Scalar]] = (
    st.sampled_from(
        [
            to_float_strategy(MIN_SCALAR, MAX_SCALAR),
            st.fractions(MIN_SCALAR, MAX_SCALAR, max_denominator=MAX_SCALAR),
        ]
    )
)


def to_point_strategy(
    scalar_strategy: st.SearchStrategy[ScalarT], /
) -> st.SearchStrategy[Point[ScalarT]]:
    return st.builds(context.point_cls, scalar_strategy, scalar_strategy)


point_strategy_strategy = scalar_strategy_strategy.map(to_point_strategy)


def to_segment_strategy(
    scalar_strategy: st.SearchStrategy[ScalarT], /
) -> st.SearchStrategy[Segment[ScalarT]]:
    return st.lists(
        to_point_strategy(scalar_strategy), min_size=2, max_size=2, unique=True
    ).map(pack(context.segment_cls))


def to_box_strategy(
    scalar_strategy: st.SearchStrategy[ScalarT], /
) -> st.SearchStrategy[Box[ScalarT]]:
    return (
        to_pairs(
            st.lists(scalar_strategy, min_size=2, max_size=2, unique=True).map(
                sorted
            )
        )
        .map(pack(add))
        .map(pack(context.box_cls))
    )
