from math import sqrt

from locus.hints import (Coordinate,
                         Interval,
                         Point)


def ceil_division(dividend: int, divisor: int) -> int:
    return -(-dividend // divisor)


HILBERT_SQUARE_SIZE = 2 ** 16
HILBERT_MAX_COORDINATE = HILBERT_SQUARE_SIZE - 1


def to_hilbert_index(x: int, y: int) -> int:
    # based on https://github.com/rawrunprotected/hilbert_curves
    assert 0 <= x <= HILBERT_MAX_COORDINATE
    assert 0 <= y <= HILBERT_MAX_COORDINATE
    a = x ^ y
    b = HILBERT_MAX_COORDINATE ^ a
    c, d = (HILBERT_MAX_COORDINATE ^ (x | y),
            x & (y ^ HILBERT_MAX_COORDINATE))
    a, b, c, d = (a | (b >> 1),
                  (a >> 1) ^ a,
                  ((c >> 1) ^ (b & (d >> 1))) ^ c,
                  ((a & (c >> 1)) ^ (d >> 1)) ^ d)
    a, b, c, d = (((a & (a >> 2)) ^ (b & (b >> 2))),
                  ((a & (b >> 2)) ^ (b & ((a ^ b) >> 2))),
                  c ^ ((a & (c >> 2)) ^ (b & (d >> 2))),
                  d ^ ((b & (c >> 2)) ^ ((a ^ b) & (d >> 2))))
    a, b, c, d = (((a & (a >> 4)) ^ (b & (b >> 4))),
                  ((a & (b >> 4)) ^ (b & ((a ^ b) >> 4))),
                  c ^ ((a & (c >> 4)) ^ (b & (d >> 4))),
                  d ^ ((b & (c >> 4)) ^ ((a ^ b) & (d >> 4))))
    c ^= ((a & (c >> 8)) ^ (b & (d >> 8)))
    d ^= ((b & (c >> 8)) ^ ((a ^ b) & (d >> 8)))
    a, b = c ^ (c >> 1), d ^ (d >> 1)
    i0 = x ^ y
    i1 = b | (HILBERT_MAX_COORDINATE ^ (i0 | a))
    return ((interleave(i1) << 1) | interleave(i0)) >> 0


def interleave(value: int) -> int:
    value = (value | (value << 8)) & 0x00FF00FF
    value = (value | (value << 4)) & 0x0F0F0F0F
    value = (value | (value << 2)) & 0x33333333
    return (value | (value << 1)) & 0x55555555


def linear_distance(left: Coordinate, right: Coordinate) -> Coordinate:
    return abs(left - right)


def planar_distance(left: Point, right: Point) -> Coordinate:
    return sqrt(sum((left_coordinate - right_coordinate) ** 2
                    for left_coordinate, right_coordinate in zip(left, right)))


def point_in_interval(point: Point, interval: Interval) -> bool:
    return all(min_coordinate <= point_coordinate <= max_coordinate
               for point_coordinate, (min_coordinate,
                                      max_coordinate) in zip(point, interval))


def distance_to_planar_interval(point: Point,
                                interval: Interval) -> Coordinate:
    x, y = point
    (min_x, max_x), (min_y, max_y) = interval
    dx = distance_to_linear_interval(x, min_x, max_x)
    dy = distance_to_linear_interval(y, min_y, max_y)
    return dx * dx + dy * dy


def distance_to_linear_interval(coordinate: Coordinate,
                                min_coordinate: Coordinate,
                                max_coordinate: Coordinate) -> Coordinate:
    return (min_coordinate - coordinate
            if coordinate < min_coordinate
            else (coordinate - max_coordinate
                  if coordinate > max_coordinate
                  else 0))
