from locus.hints import (Coordinate,
                         Interval,
                         Point)


def squared_distance(left: Point, right: Point) -> Coordinate:
    return sum((left_coordinate - right_coordinate) ** 2
               for left_coordinate, right_coordinate in zip(left, right))


def point_in_interval(point: Point, interval: Interval) -> bool:
    return all(min_coordinate <= point_coordinate <= max_coordinate
               for point_coordinate, (min_coordinate,
                                      max_coordinate) in zip(point, interval))
