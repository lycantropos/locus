from collections.abc import Callable, Sequence
from functools import reduce
from math import floor, inf
from typing import Generic, TypeAlias, cast

from ground.hints import Box, Point, Segment
from reprit.base import generate_repr
from typing_extensions import Self

from . import hilbert
from .hints import HasCustomRepr, ScalarT
from .utils import ceil_division

Item: TypeAlias = tuple[int, Segment[ScalarT]]


class Node(HasCustomRepr, Generic[ScalarT]):
    """Represents node of segmental *R*-tree."""

    __slots__ = (
        'box',
        'box_point_metric',
        'box_segment_metric',
        'children',
        'index',
        'segment',
        'segment_point_metric',
        'segments_metric',
    )

    def __init__(
        self,
        index: int,
        box: Box[ScalarT],
        segment: Segment[ScalarT] | None,
        children: Sequence[Self] | None,
        box_point_metric: Callable[[Box[ScalarT], Point[ScalarT]], ScalarT],
        box_segment_metric: Callable[
            [Box[ScalarT], Segment[ScalarT]], ScalarT
        ],
        segment_point_metric: Callable[
            [Segment[ScalarT], Point[ScalarT]], ScalarT
        ],
        segments_metric: Callable[
            [Segment[ScalarT], Segment[ScalarT]], ScalarT
        ],
    ) -> None:
        self.box, self.children, self.index, self.segment = (
            box,
            children,
            index,
            segment,
        )
        (
            self.box_point_metric,
            self.box_segment_metric,
            self.segment_point_metric,
            self.segments_metric,
        ) = (
            box_point_metric,
            box_segment_metric,
            segment_point_metric,
            segments_metric,
        )

    __repr__ = generate_repr(__init__)

    @property
    def is_leaf(self, /) -> bool:
        return self.children is None

    @property
    def item(self, /) -> Item[ScalarT]:
        assert self.segment is not None, self
        return self.index, self.segment

    def distance_to_point(self, point: Point[ScalarT], /) -> ScalarT:
        if self.is_leaf:
            assert self.segment is not None, self
            return (
                self.segment_point_metric(self.segment, point)
                or self._minus_inf()
            )
        return self.box_point_metric(self.box, point)

    def distance_to_segment(self, segment: Segment[ScalarT], /) -> ScalarT:
        if self.is_leaf:
            assert self.segment is not None, self
            return (
                self.segments_metric(self.segment, segment)
                or self._minus_inf()
            )
        return self.box_segment_metric(self.box, segment)

    @classmethod
    def _minus_inf(cls, /) -> ScalarT:
        return cast(ScalarT, -inf)


def create_root(
    segments: Sequence[Segment[ScalarT]],
    boxes: Sequence[Box[ScalarT]],
    max_children: int,
    boxes_merger: Callable[[Box[ScalarT], Box[ScalarT]], Box[ScalarT]],
    box_point_metric: Callable[[Box[ScalarT], Point[ScalarT]], ScalarT],
    box_segment_metric: Callable[[Box[ScalarT], Segment[ScalarT]], ScalarT],
    segment_point_metric: Callable[
        [Segment[ScalarT], Point[ScalarT]], ScalarT
    ],
    segments_metric: Callable[[Segment[ScalarT], Segment[ScalarT]], ScalarT],
    coordinate_factory: Callable[[int], ScalarT],
) -> Node[ScalarT]:
    nodes = [
        Node(
            index,
            box,
            segment,
            None,
            box_point_metric,
            box_segment_metric,
            segment_point_metric,
            segments_metric,
        )
        for index, (box, segment) in enumerate(
            zip(boxes, segments, strict=True)
        )
    ]
    root_box = reduce(boxes_merger, boxes)
    leaves_count = len(nodes)
    if leaves_count <= max_children:
        # only one node, skip sorting and just fill the root box
        return Node(
            len(nodes),
            root_box,
            None,
            nodes,
            box_point_metric,
            box_segment_metric,
            segment_point_metric,
            segments_metric,
        )

    one = coordinate_factory(1)
    two = coordinate_factory(2)

    def node_key(
        node: Node[ScalarT],
        double_root_delta_x: ScalarT = (
            two * (root_box.max_x - root_box.min_x) or one
        ),
        double_root_delta_y: ScalarT = (
            two * (root_box.max_y - root_box.min_y) or one
        ),
        double_root_min_x: ScalarT = two * root_box.min_x,
        double_root_min_y: ScalarT = two * root_box.min_y,
    ) -> int:
        box = node.box
        return hilbert.index(
            floor(
                (
                    coordinate_factory(hilbert.MAX_COORDINATE)
                    * (box.min_x + box.max_x - double_root_min_x)
                )
                / double_root_delta_x
            ),
            floor(
                (
                    coordinate_factory(hilbert.MAX_COORDINATE)
                    * (box.min_y + box.max_y - double_root_min_y)
                )
                / double_root_delta_y
            ),
        )

    nodes = sorted(nodes, key=node_key)
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
            nodes.append(
                Node(
                    len(nodes),
                    reduce(boxes_merger, [child.box for child in children]),
                    None,
                    children,
                    box_point_metric,
                    box_segment_metric,
                    segment_point_metric,
                    segments_metric,
                )
            )
            start = stop
    return nodes[-1]
