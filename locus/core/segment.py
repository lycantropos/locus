from typing import Type

from ground.hints import (Coordinate,
                          Point,
                          Segment)
from ground.linear import SegmentsRelationship

from locus.hints import Interval
from . import interval as _interval
from .hints import (Divider,
                    DotProducer,
                    SegmentsRelater,
                    SquareRooter)


def distance_to(divider: Divider,
                dot_producer: DotProducer,
                segments_relater: SegmentsRelater,
                square_rooter: SquareRooter,
                first_start: Point,
                first_end: Point,
                second_start: Point,
                second_end: Point) -> Coordinate:
    return square_rooter(_squared_distance_to(divider, dot_producer,
                                              segments_relater, first_start,
                                              first_end, second_start,
                                              second_end))


def distance_to_interval(divider: Divider,
                         dot_producer: DotProducer,
                         point_cls: Type[Point],
                         segments_relater: SegmentsRelater,
                         square_rooter: SquareRooter,
                         segment: Segment,
                         interval: Interval) -> Coordinate:
    start, end = segment.start, segment.end
    if (_interval.contains_point(interval, start)
            or _interval.contains_point(interval, end)):
        return 0
    (min_x, max_x), (min_y, max_y) = interval
    return (distance_to(divider, dot_producer, segments_relater, square_rooter,
                        point_cls(min_x, min_y), point_cls(min_x, max_y),
                        start, end)
            if min_x == max_x
            else
            (distance_to(divider, dot_producer, segments_relater,
                         square_rooter, point_cls(min_x, min_y),
                         point_cls(max_x, min_y), start, end)
             if min_y == max_y
             else _distance_to_non_degenerate_interval(
                    divider, dot_producer, point_cls, segments_relater,
                    square_rooter, start, end, max_x, max_y, min_x, min_y)))


def distance_to_point(divider: Divider,
                      dot_producer: DotProducer,
                      square_rooter: SquareRooter,
                      segment: Segment,
                      point: Point) -> Coordinate:
    return square_rooter(_squared_distance_to_point(
            divider, dot_producer, segment.start, segment.end, point))


def _distance_to_non_degenerate_interval(divider: Divider,
                                         dot_producer: DotProducer,
                                         point_cls: Type[Point],
                                         segments_relater: SegmentsRelater,
                                         square_rooter: SquareRooter,
                                         start: Point,
                                         end: Point,
                                         max_x: Coordinate,
                                         max_y: Coordinate,
                                         min_x: Coordinate,
                                         min_y: Coordinate) -> Coordinate:
    bottom_left = point_cls(min_x, min_y)
    bottom_right = point_cls(max_x, min_y)
    bottom_side_distance = _squared_distance_to(divider, dot_producer,
                                                segments_relater, start, end,
                                                bottom_left, bottom_right)
    if not bottom_side_distance:
        return bottom_side_distance
    top_right = point_cls(max_x, max_y)
    right_side_distance = _squared_distance_to(divider, dot_producer,
                                               segments_relater, start, end,
                                               bottom_right, top_right)
    if not right_side_distance:
        return right_side_distance
    top_left = point_cls(min_x, max_y)
    top_side_distance = _squared_distance_to(divider, dot_producer,
                                             segments_relater, start, end,
                                             top_left, top_right)
    if not top_side_distance:
        return top_side_distance
    left_side_distance = _squared_distance_to(divider, dot_producer,
                                              segments_relater, start, end,
                                              bottom_left, top_left)
    return (square_rooter(min(bottom_side_distance, right_side_distance,
                              left_side_distance, top_side_distance))
            if left_side_distance
            else left_side_distance)


def _squared_distance_to(divider: Divider,
                         dot_producer: DotProducer,
                         segments_relater: SegmentsRelater,
                         first_start: Point,
                         first_end: Point,
                         second_start: Point,
                         second_end: Point) -> Coordinate:
    return (min(_squared_distance_to_point(divider, dot_producer, first_start,
                                           first_end, second_start),
                _squared_distance_to_point(divider, dot_producer, first_start,
                                           first_end, second_end),
                _squared_distance_to_point(divider, dot_producer, second_start,
                                           second_end, first_start),
                _squared_distance_to_point(divider, dot_producer, second_start,
                                           second_end, first_end))
            if (segments_relater(first_start, first_end, second_start,
                                 second_end)
                is SegmentsRelationship.NONE)
            else 0)


def _squared_distance_to_point(divider: Divider,
                               dot_producer: DotProducer,
                               start: Point,
                               end: Point,
                               point: Point) -> Coordinate:
    factor = max(0, min(1, divider(dot_producer(start, point, start, end),
                                   dot_producer(start, end, start, end))))
    return (((start.x - point.x) + factor * (end.x - start.x)) ** 2
            + ((start.y - point.y) + factor * (end.y - start.y)) ** 2)
