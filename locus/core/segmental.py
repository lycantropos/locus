from math import inf
from typing import (Callable,
                    Optional,
                    Sequence,
                    Tuple)

from ground.hints import (Box,
                          Coordinate,
                          Point,
                          Segment)
from reprit.base import generate_repr

Item = Tuple[int, Segment]


class Node:
    """Represents node of segmental *R*-tree."""
    __slots__ = ('box', 'box_point_metric', 'box_segment_metric',
                 'children', 'index', 'segment', 'segment_point_metric',
                 'segments_metric')

    def __init__(self,
                 index: int,
                 box: Box,
                 segment: Optional[Segment],
                 children: Optional[Sequence['Node']],
                 box_point_metric: Callable[[Box, Point], Coordinate],
                 box_segment_metric
                 : Callable[[Box, Point, Point], Coordinate],
                 segment_point_metric
                 : Callable[[Point, Point, Point], Coordinate],
                 segments_metric
                 : Callable[[Point, Point, Point, Point], Coordinate]
                 ) -> None:
        self.box, self.children, self.index, self.segment = (
            box, children, index, segment)
        (self.box_point_metric, self.box_segment_metric,
         self.segment_point_metric, self.segments_metric) = (
            box_point_metric, box_segment_metric, segment_point_metric,
            segments_metric)

    __repr__ = generate_repr(__init__)

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    @property
    def item(self) -> Item:
        return self.index, self.segment

    def distance_to_point(self,
                          point: Point,
                          *,
                          _minus_inf: Coordinate = -inf) -> Coordinate:
        return (self.segment_point_metric(self.segment.start, self.segment.end,
                                          point) or _minus_inf
                if self.is_leaf
                else self.box_point_metric(self.box, point))

    def distance_to_segment(self,
                            segment: Segment,
                            *,
                            _minus_inf: Coordinate = -inf) -> Coordinate:
        return (self.segments_metric(self.segment.start, self.segment.end,
                                     segment.start, segment.end) or _minus_inf
                if self.is_leaf
                else self.box_segment_metric(self.box, segment.start,
                                             segment.end))
