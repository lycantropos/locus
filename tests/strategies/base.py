import sys
from decimal import Decimal
from fractions import Fraction
from functools import partial
from operator import add

from ground.hints import Coordinate
from hypothesis import strategies

from tests.utils import (Box,
                         Point,
                         Segment,
                         Strategy,
                         pack,
                         to_pairs)

MAX_COORDINATE = 10 ** 15
MIN_COORDINATE = -MAX_COORDINATE


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


def coordinates_to_points(coordinates: Strategy[Coordinate]
                          ) -> Strategy[Point]:
    return strategies.builds(Point, coordinates, coordinates)


points_strategies = coordinates_strategies.map(coordinates_to_points)


def coordinates_to_segments(coordinates: Strategy[Coordinate]
                            ) -> Strategy[Segment]:
    return (strategies.lists(coordinates_to_points(coordinates),
                             min_size=2,
                             max_size=2,
                             unique=True)
            .map(pack(Segment)))


def coordinates_to_boxes(coordinates: Strategy[Coordinate]
                             ) -> Strategy[Box]:
    return (to_pairs(strategies.lists(coordinates,
                                      min_size=2,
                                      max_size=2,
                                      unique=True)
                     .map(sorted))
            .map(pack(add))
            .map(pack(Box)))
