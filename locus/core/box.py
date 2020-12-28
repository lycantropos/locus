from typing import Type

from ground.hints import (Box,
                          Coordinate,
                          Point)


def contains_point(box: Box, point: Point) -> bool:
    return (box.min_x <= point.x <= box.max_x
            and box.min_y <= point.y <= box.max_y)


def distance_to_point(box: Box, point: Point) -> Coordinate:
    return (_distance_to_linear_interval(point.x, box.min_x, box.max_x) ** 2
            + _distance_to_linear_interval(point.y, box.min_y, box.max_y) ** 2)


def _distance_to_linear_interval(coordinate: Coordinate,
                                 min_coordinate: Coordinate,
                                 max_coordinate: Coordinate) -> Coordinate:
    return (min_coordinate - coordinate
            if coordinate < min_coordinate
            else (coordinate - max_coordinate
                  if coordinate > max_coordinate
                  else 0))


def overlaps(left: Box, right: Box) -> bool:
    return (right.min_x < left.max_x and left.min_x < right.max_x
            and right.min_y < left.max_y and left.min_y < right.max_y)


def is_subset_of(test: Box, goal: Box) -> bool:
    return (goal.min_x <= test.min_x and test.max_x <= goal.max_x
            and goal.min_y <= test.min_y and test.max_y <= goal.max_y)


def merge(box_cls: Type[Box], left: Box, right: Box) -> Box:
    return box_cls(min(left.min_x, right.min_x), max(left.max_x, right.max_x),
                   min(left.min_y, right.min_y), max(left.max_y, right.max_y))
