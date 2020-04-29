from math import sqrt

from locus.hints import (Coordinate,
                         Point)


def ceil_division(dividend: int, divisor: int) -> int:
    return -(-dividend // divisor)


def linear_distance(left: Coordinate, right: Coordinate) -> Coordinate:
    return abs(left - right)


def planar_distance(left: Point, right: Point) -> Coordinate:
    return sqrt(sum((left_coordinate - right_coordinate) ** 2
                    for left_coordinate, right_coordinate in zip(left, right)))
