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


def distance_to_interval(segment: Segment, interval: Interval) -> Coordinate:
    start, end = segment
    if (_interval.contains_point(interval, start)
            or _interval.contains_point(interval, end)):
        return 0
    (min_x, max_x), (min_y, max_y) = interval
    if min_x == max_x:
        return distance_to(((min_x, min_y), (min_x, max_y)), segment)
    elif min_y == max_y:
        return distance_to(((min_x, min_y), (max_x, min_y)), segment)
    bottom_left = min_x, min_y
    bottom_right = max_x, min_y
    bottom_side_distance = squared_distance_to(segment,
                                               (bottom_left, bottom_right))
    if not bottom_side_distance:
        return bottom_side_distance
    top_right = max_x, max_y
    right_side_distance = squared_distance_to(segment,
                                              (bottom_right, top_right))
    if not right_side_distance:
        return right_side_distance
    top_left = min_x, max_y
    top_side_distance = squared_distance_to(segment, (top_left, top_right))
    if not top_side_distance:
        return top_side_distance
    left_side_distance = squared_distance_to(segment, (bottom_left, top_left))
    return (sqrt(min(bottom_side_distance, right_side_distance,
                     left_side_distance, top_side_distance))
            if left_side_distance
            else left_side_distance)


def distance_to(left: Segment, right: Segment) -> Coordinate:
    return sqrt(squared_distance_to(left, right))


def distance_to_point(segment: Segment, point: Point) -> Coordinate:
    return sqrt(squared_distance_to_point(segment, point))


def squared_distance_to(left: Segment, right: Segment) -> Coordinate:
    left_start, left_end = left
    right_start, right_end = right
    return (min(squared_distance_to_point(left, right_start),
                squared_distance_to_point(left, right_end),
                squared_distance_to_point(right, left_start),
                squared_distance_to_point(right, left_end))
            if segments_relationship(left, right) is SegmentsRelationship.NONE
            else 0)


def squared_distance_to_point(segment: Segment, point: Point) -> Coordinate:
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


def _robust_divide(dividend: Coordinate, divisor: Coordinate) -> Coordinate:
    return (Fraction(dividend, divisor)
            if isinstance(dividend, int)
            else dividend / divisor)
