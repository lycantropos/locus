from operator import attrgetter
from typing import (Callable,
                    Tuple,
                    Union)

from ground.hints import (Coordinate,
                          Point)
from reprit.base import generate_repr

Item = Tuple[int, Point]
NIL = None
PROJECTORS = attrgetter('x'), attrgetter('y')


class Node:
    """Represents node of *kd*-tree."""
    __slots__ = ('index', 'is_y_axis', 'left', 'metric', 'point', 'projector',
                 'right')

    def __init__(self,
                 index: int,
                 point: Point,
                 is_y_axis: bool,
                 left: Union['Node', NIL],
                 right: Union['Node', NIL],
                 metric: Callable[[Point, Point], Coordinate]) -> None:
        self.index, self.point = index, point
        self.is_y_axis, self.projector = is_y_axis, PROJECTORS[is_y_axis]
        self.left, self.right = left, right
        self.metric = metric

    __repr__ = generate_repr(__init__)

    @property
    def item(self) -> Item:
        """Returns item of the node."""
        return self.index, self.point

    @property
    def projection(self) -> Coordinate:
        """Returns projection of the node point onto the corresponding axis."""
        return self.projector(self.point)

    def distance_to_point(self, point: Point) -> Coordinate:
        """Calculates distance to given point."""
        return self.metric(self.point, point)

    def distance_to_coordinate(self, coordinate: Coordinate) -> Coordinate:
        """Calculates distance to given coordinate."""
        return (self.projection - coordinate) ** 2
