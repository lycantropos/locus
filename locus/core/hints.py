from typing import Callable

from ground.hints import (Coordinate,
                          Point)
from ground.linear import SegmentsRelationship

Divider = Callable[[Coordinate, Coordinate], Coordinate]
DotProducer = Callable[[Point, Point, Point, Point], Coordinate]
SegmentsRelater = Callable[[Point, Point, Point, Point], SegmentsRelationship]
SquareRooter = Callable[[Coordinate], Coordinate]
