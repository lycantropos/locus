from math import hypot

from locus.hints import (Coordinate,
                         Interval,
                         Point)


def contains_point(interval: Interval, point: Point) -> bool:
    return all(min_coordinate <= point_coordinate <= max_coordinate
               for point_coordinate, (min_coordinate,
                                      max_coordinate) in zip(point, interval))


def planar_distance_to_point(interval: Interval, point: Point) -> Coordinate:
    x, y = point
    (min_x, max_x), (min_y, max_y) = interval
    return hypot(_distance_to_linear_interval(x, min_x, max_x),
                 _distance_to_linear_interval(y, min_y, max_y))


def _distance_to_linear_interval(coordinate: Coordinate,
                                 min_coordinate: Coordinate,
                                 max_coordinate: Coordinate) -> Coordinate:
    return (min_coordinate - coordinate
            if coordinate < min_coordinate
            else (coordinate - max_coordinate
                  if coordinate > max_coordinate
                  else 0))


def overlaps(left: Interval, right: Interval) -> bool:
    (left_x_min, left_x_max), (left_y_min, left_y_max) = left
    (right_x_min, right_x_max), (right_y_min, right_y_max) = right
    return (right_x_min < left_x_max and left_x_min < right_x_max
            and right_y_min < left_y_max and left_y_min < right_y_max)


def is_subset_of(test: Interval, goal: Interval) -> bool:
    (goal_x_min, goal_x_max), (goal_y_min, goal_y_max) = goal
    (test_x_min, test_x_max), (test_y_min, test_y_max) = test
    return (goal_x_min <= test_x_min and test_x_max <= goal_x_max
            and goal_y_min <= test_y_min and test_y_max <= goal_y_max)


def merge(left: Interval, right: Interval) -> Interval:
    (left_x_min, left_x_max), (left_y_min, left_y_max) = left
    (right_x_min, right_x_max), (right_y_min, right_y_max) = right
    return ((min(left_x_min, right_x_min), max(left_x_max, right_x_max)),
            (min(left_y_min, right_y_min), max(left_y_max, right_y_max)))
