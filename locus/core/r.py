from typing import (Callable,
                    Optional,
                    Sequence,
                    Tuple)

from ground.hints import (Box,
                          Coordinate,
                          Point)
from reprit.base import generate_repr

Item = Tuple[int, Box]


class Node:
    """Represents node of *R*-tree."""
    __slots__ = 'box', 'children', 'index', 'metric'

    def __init__(self,
                 index: int,
                 box: Box,
                 children: Optional[Sequence['Node']],
                 metric: Callable[[Box, Point], Coordinate]) -> None:
        self.box, self.children, self.index, self.metric = (box, children,
                                                            index, metric)

    __repr__ = generate_repr(__init__)

    @property
    def is_leaf(self) -> bool:
        """Checks whether the node is a leaf."""
        return self.children is None

    @property
    def item(self) -> Item:
        """Returns underlying index with box."""
        return self.index, self.box

    def distance_to_point(self, point: Point) -> Coordinate:
        """Calculates distance to given point."""
        return self.metric(self.box, point)
