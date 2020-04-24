from locus.hints import (Coordinate,
                         Point)


def squared_distance(left: Point, right: Point) -> Coordinate:
    return sum((left_coordinate - right_coordinate) ** 2
               for left_coordinate, right_coordinate in zip(left, right))
