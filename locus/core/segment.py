from fractions import Fraction
from math import sqrt

from robust import projection
from robust.linear import (SegmentsRelationship,
                           segments_relationship)

from locus.hints import (Coordinate,
                         Interval,
                         Point,
                         Segment)
from . import interval as _interval


def distance_to(left: Segment, right: Segment) -> Coordinate:
    return sqrt(_squared_distance_to(left, right))


def distance_to_interval(segment: Segment, interval: Interval) -> Coordinate:
    start, end = segment
    if (_interval.contains_point(interval, start)
            or _interval.contains_point(interval, end)):
        return 0
    (min_x, max_x), (min_y, max_y) = interval
    return (distance_to(((min_x, min_y), (min_x, max_y)), segment)
            if min_x == max_x
            else (distance_to(((min_x, min_y), (max_x, min_y)), segment)
                  if min_y == max_y
                  else _distance_to_non_degenerate_interval(segment, max_x,
                                                            max_y, min_x,
                                                            min_y)))


def distance_to_point(segment: Segment, point: Point) -> Coordinate:
    return sqrt(_squared_distance_to_point(segment, point))


def _distance_to_non_degenerate_interval(segment: Segment,
                                         max_x: Coordinate,
                                         max_y: Coordinate,
                                         min_x: Coordinate,
                                         min_y: Coordinate) -> Coordinate:
    bottom_left = min_x, min_y
    bottom_right = max_x, min_y
    bottom_side_distance = _squared_distance_to(segment,
                                                (bottom_left, bottom_right))
    if not bottom_side_distance:
        return bottom_side_distance
    top_right = max_x, max_y
    right_side_distance = _squared_distance_to(segment,
                                               (bottom_right, top_right))
    if not right_side_distance:
        return right_side_distance
    top_left = min_x, max_y
    top_side_distance = _squared_distance_to(segment, (top_left, top_right))
    if not top_side_distance:
        return top_side_distance
    left_side_distance = _squared_distance_to(segment, (bottom_left, top_left))
    return (sqrt(min(bottom_side_distance, right_side_distance,
                     left_side_distance, top_side_distance))
            if left_side_distance
            else left_side_distance)


def _robust_divide(dividend: Coordinate, divisor: Coordinate) -> Coordinate:
    return (Fraction(dividend, divisor)
            if isinstance(dividend, int)
            else dividend / divisor)


def _squared_distance_to(left: Segment, right: Segment) -> Coordinate:
    left_start, left_end = left
    right_start, right_end = right
    return (min(_squared_distance_to_point(left, right_start),
                _squared_distance_to_point(left, right_end),
                _squared_distance_to_point(right, left_start),
                _squared_distance_to_point(right, left_end))
            if segments_relationship(left, right) is SegmentsRelationship.NONE
            else 0)


def _squared_distance_to_point(segment: Segment, point: Point) -> Coordinate:
    start, end = segment
    factor = max(0, min(1, _robust_divide(projection.signed_length(
            start, point, start, end),
            _squared_points_distance(end, start))))
    start_x, start_y = start
    end_x, end_y = end
    return _squared_points_distance((start_x + factor * (end_x - start_x),
                                     start_y + factor * (end_y - start_y)),
                                    point)


def _squared_points_distance(left: Point, right: Point) -> Coordinate:
    (left_x, left_y), (right_x, right_y) = left, right
    return (left_x - right_x) ** 2 + (left_y - right_y) ** 2
