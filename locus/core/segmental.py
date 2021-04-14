from functools import reduce
from math import (floor,
                  inf)
from typing import (Callable,
                    Optional,
                    Sequence,
                    Tuple)

from ground.hints import (Box,
                          Coordinate,
                          Point,
                          Segment)
from reprit.base import generate_repr

from . import hilbert
from .utils import ceil_division

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


def create_root(segments: Sequence[Segment],
                boxes: Sequence[Box],
                max_children: int,
                boxes_merger: Callable[[Box, Box], Box],
                box_point_metric: Callable[[Box, Point], Coordinate],
                box_segment_metric: Callable[[Box, Point, Point], Coordinate],
                segment_point_metric
                : Callable[[Point, Point, Point], Coordinate],
                segments_metric
                : Callable[[Point, Point, Point, Point], Coordinate]) -> Node:
    nodes = [Node(index, box, segment, None, box_point_metric,
                  box_segment_metric, segment_point_metric, segments_metric)
             for index, (box, segment) in enumerate(zip(boxes, segments))]
    root_box = reduce(boxes_merger, boxes)
    leaves_count = len(nodes)
    if leaves_count <= max_children:
        # only one node, skip sorting and just fill the root box
        return Node(len(nodes), root_box, None, nodes, box_point_metric,
                    box_segment_metric, segment_point_metric, segments_metric)
    else:
        def node_key(node: Node,
                     double_root_delta_x: Coordinate
                     = 2 * (root_box.max_x - root_box.min_x) or 1,
                     double_root_delta_y: Coordinate
                     = 2 * (root_box.max_y - root_box.min_y) or 1,
                     double_root_min_x: Coordinate = 2 * root_box.min_x,
                     double_root_min_y: Coordinate = 2 * root_box.min_y
                     ) -> int:
            box = node.box
            return hilbert.index(floor(hilbert.MAX_COORDINATE
                                       * (box.min_x + box.max_x
                                          - double_root_min_x)
                                       / double_root_delta_x),
                                 floor(hilbert.MAX_COORDINATE
                                       * (box.min_y + box.max_y
                                          - double_root_min_y)
                                       / double_root_delta_y))

        nodes = sorted(nodes,
                       key=node_key)
        nodes_count = step = leaves_count
        levels_limits = [nodes_count]
        while True:
            step = ceil_division(step, max_children)
            if step == 1:
                break
            nodes_count += step
            levels_limits.append(nodes_count)
        start = 0
        for level_limit in levels_limits:
            while start < level_limit:
                stop = min(start + max_children, level_limit)
                children = nodes[start:stop]
                nodes.append(Node(len(nodes),
                                  reduce(boxes_merger,
                                         [child.box for child in children]),
                                  None, children, box_point_metric,
                                  box_segment_metric, segment_point_metric,
                                  segments_metric))
                start = stop
        return nodes[-1]
