from math import sqrt

from locus.hints import (Coordinate,
                         Interval,
                         Point)


def linear_distance(left: Coordinate, right: Coordinate) -> Coordinate:
    return abs(left - right)


def planar_distance(left: Point, right: Point) -> Coordinate:
    return sqrt(sum((left_coordinate - right_coordinate) ** 2
                    for left_coordinate, right_coordinate in zip(left, right)))


def point_in_interval(point: Point, interval: Interval) -> bool:
    return all(min_coordinate <= point_coordinate <= max_coordinate
               for point_coordinate, (min_coordinate,
                                      max_coordinate) in zip(point, interval))
