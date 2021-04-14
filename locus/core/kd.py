from operator import attrgetter
from typing import (Callable,
                    Sequence,
                    Tuple,
                    Type,
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


def create_node(cls: Type[Node],
                indices: Sequence[int],
                points: Sequence[Point],
                is_y_axis: bool,
                metric: Callable[[Point, Point], Coordinate]
                ) -> Union[Node, NIL]:
    if not indices:
        return NIL
    indices = sorted(indices,
                     key=(lambda index, projector=PROJECTORS[is_y_axis]
                          : projector(points[index])))
    middle_index = (len(indices) - 1) // 2
    pivot_index = indices[middle_index]
    next_is_y_axis = not is_y_axis
    return cls(pivot_index, points[pivot_index], is_y_axis,
               create_node(cls, indices[:middle_index], points,
                           next_is_y_axis, metric),
               create_node(cls, indices[middle_index + 1:], points,
                           next_is_y_axis, metric), metric)
