from ground.hints import (Coordinate,
                          Point)

from .hints import SquareRooter


def ceil_division(dividend: int, divisor: int) -> int:
    return -(-dividend // divisor)


def points_distance(square_rooter: SquareRooter,
                    first: Point,
                    second: Point) -> Coordinate:
    return square_rooter((second.x - first.x) ** 2 + (second.y - first.y) ** 2)
