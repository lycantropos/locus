import sys
from decimal import Decimal
from fractions import Fraction
from functools import partial

from hypothesis import strategies

from locus.hints import (Coordinate,
                         Interval,
                         Point)
from tests.bounds import (MAX_AXES_COUNT,
                          MAX_COORDINATE,
                          MIN_COORDINATE)
from tests.utils import (Strategy,
                         to_homogeneous_tuples)


def to_floats(min_value: Coordinate,
              max_value: Coordinate,
              *,
              allow_nan: bool = False,
              allow_infinity: bool = False) -> Strategy:
    return (strategies.floats(min_value=min_value,
                              max_value=max_value,
                              allow_nan=allow_nan,
                              allow_infinity=allow_infinity)
            .map(to_digits_count))


def to_digits_count(number: float,
                    *,
                    max_digits_count: int = sys.float_info.dig) -> float:
    decimal = Decimal(number).normalize()
    _, significant_digits, exponent = decimal.as_tuple()
    significant_digits_count = len(significant_digits)
    if exponent < 0:
        fixed_digits_count = (1 - exponent
                              if exponent <= -significant_digits_count
                              else significant_digits_count)
    else:
        fixed_digits_count = exponent + significant_digits_count
    if fixed_digits_count <= max_digits_count:
        return number
    whole_digits_count = max(significant_digits_count + exponent, 0)
    if whole_digits_count:
        whole_digits_offset = max(whole_digits_count - max_digits_count, 0)
        decimal /= 10 ** whole_digits_offset
        whole_digits_count -= whole_digits_offset
    else:
        decimal *= 10 ** (-exponent - significant_digits_count)
        whole_digits_count = 1
    decimal = round(decimal, max(max_digits_count - whole_digits_count, 0))
    return float(str(decimal))


coordinates_strategies_factories = {
    float: to_floats,
    Fraction: partial(strategies.fractions,
                      max_denominator=MAX_COORDINATE),
    int: strategies.integers}
coordinates_strategies = strategies.sampled_from(
        [factory(MIN_COORDINATE, MAX_COORDINATE)
         for factory in coordinates_strategies_factories.values()])
axes = strategies.integers(1, MAX_AXES_COUNT)


def coordinates_to_points(coordinates: Strategy[Coordinate],
                          *,
                          dimension: int) -> Strategy[Point]:
    return to_homogeneous_tuples(coordinates,
                                 size=dimension)


points_strategies = strategies.builds(coordinates_to_points,
                                      coordinates_strategies,
                                      dimension=axes)


def coordinates_to_intervals(coordinates: Strategy[Coordinate],
                             *,
                             dimension: int) -> Strategy[Interval]:
    return to_homogeneous_tuples(strategies.lists(coordinates,
                                                  min_size=2,
                                                  max_size=2,
                                                  unique=True)
                                 .map(sorted)
                                 .map(tuple),
                                 size=dimension)
