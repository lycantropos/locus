from fractions import Fraction
from typing import (Callable,
                    Type,
                    TypeVar)

from ground.base import Relation
from ground.hints import (Box,
                          Coordinate,
                          Point,
                          Segment)

from . import box as _box

Range = TypeVar('Range')
QuaternaryPointFunction = Callable[[Point, Point, Point, Point], Range]


def distance_to(dot_product: QuaternaryPointFunction[Coordinate],
                segments_relater: QuaternaryPointFunction[Relation],
                first_start: Point,
                first_end: Point,
                second_start: Point,
                second_end: Point) -> Coordinate:
    return (min(distance_to_point(dot_product, first_start, first_end,
                                  second_start),
                distance_to_point(dot_product, first_start, first_end,
                                  second_end),
                distance_to_point(dot_product, second_start, second_end,
                                  first_start),
                distance_to_point(dot_product, second_start, second_end,
                                  first_end))
            if (segments_relater(first_start, first_end, second_start,
                                 second_end)
                is Relation.DISJOINT)
            else 0)


def distance_to_box(dot_product: QuaternaryPointFunction[Coordinate],
                    point_cls: Type[Point],
                    segments_relationship
                    : QuaternaryPointFunction[Relation],
                    segment: Segment,
                    box: Box) -> Coordinate:
    start, end = segment.start, segment.end
    if _box.contains_point(box, start) or _box.contains_point(box, end):
        return 0
    return ((distance_to_point(dot_product, start, end,
                               point_cls(box.min_x, box.min_y))
             if box.min_y == box.max_y
             else distance_to(dot_product, segments_relationship, start, end,
                              point_cls(box.min_x, box.min_y),
                              point_cls(box.min_x, box.max_y)))
            if box.min_x == box.max_x
            else (distance_to(dot_product, segments_relationship, start, end,
                              point_cls(box.min_x, box.min_y),
                              point_cls(box.max_x, box.min_y))
                  if box.min_y == box.max_y
                  else _distance_to_non_degenerate_box(dot_product, point_cls,
                                                       segments_relationship,
                                                       start, end, box)))


def distance_to_point(dot_product: QuaternaryPointFunction[Coordinate],
                      start: Point,
                      end: Point,
                      point: Point) -> Coordinate:
    factor = max(0, min(1, (Fraction(dot_product(start, point, start, end))
                            / dot_product(start, end, start, end))))
    return (((start.x - point.x) + factor * (end.x - start.x)) ** 2
            + ((start.y - point.y) + factor * (end.y - start.y)) ** 2)


def _distance_to_non_degenerate_box(
        dot_product: QuaternaryPointFunction[Coordinate],
        point_cls: Type[Point],
        segments_relater: QuaternaryPointFunction[Relation],
        start: Point,
        end: Point,
        box: Box) -> Coordinate:
    bottom_left = point_cls(box.min_x, box.min_y)
    bottom_right = point_cls(box.max_x, box.min_y)
    bottom_side_distance = distance_to(dot_product, segments_relater, start,
                                       end, bottom_left, bottom_right)
    if not bottom_side_distance:
        return bottom_side_distance
    top_right = point_cls(box.max_x, box.max_y)
    right_side_distance = distance_to(dot_product, segments_relater, start,
                                      end, bottom_right, top_right)
    if not right_side_distance:
        return right_side_distance
    top_left = point_cls(box.min_x, box.max_y)
    top_side_distance = distance_to(dot_product, segments_relater, start, end,
                                    top_left, top_right)
    if not top_side_distance:
        return top_side_distance
    left_side_distance = distance_to(dot_product, segments_relater, start, end,
                                     bottom_left, top_left)
    return (min(bottom_side_distance, right_side_distance, left_side_distance,
                top_side_distance)
            if left_side_distance
            else left_side_distance)
