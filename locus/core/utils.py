from ground.hints import (Coordinate,
                          Point)


def ceil_division(dividend: int, divisor: int) -> int:
    return -(-dividend // divisor)


def points_distance(first: Point, second: Point) -> Coordinate:
    return (second.x - first.x) ** 2 + (second.y - first.y) ** 2
