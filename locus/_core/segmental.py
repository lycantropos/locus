from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import reduce
from math import floor, inf
from typing import Generic, TypeAlias, cast

from ground.hints import Box, Point, Segment
from reprit.base import generate_repr
from typing_extensions import Self, TypeIs

from . import hilbert
from .hints import HasCustomRepr, ScalarT
from .utils import ceil_division

Item: TypeAlias = tuple[int, Segment[ScalarT]]
_BoxSegmentMetric: TypeAlias = Callable[
    [Box[ScalarT], Segment[ScalarT]], ScalarT
]
_BoxPointMetric: TypeAlias = Callable[[Box[ScalarT], Point[ScalarT]], ScalarT]
_CoordinateFactory: TypeAlias = Callable[[int], ScalarT]
_SegmentPointMetric: TypeAlias = Callable[
    [Segment[ScalarT], Point[ScalarT]], ScalarT
]
_SegmentsMetric: TypeAlias = Callable[
    [Segment[ScalarT], Segment[ScalarT]], ScalarT
]


class Leaf(HasCustomRepr, Generic[ScalarT]):
    """Represents leaf of segmental *R*-tree."""

    @property
    def box(self, /) -> Box[ScalarT]:
        return self._box

    @property
    def index(self, /) -> int:
        return self._index

    __slots__ = (
        '_box',
        '_index',
        '_segment',
        '_segment_point_metric',
        '_segments_metric',
        '_zero',
    )

    def __init__(
        self,
        index: int,
        box: Box[ScalarT],
        _segment: Segment[ScalarT],
        _segment_point_metric: _SegmentPointMetric[ScalarT],
        _segments_metric: _SegmentsMetric[ScalarT],
        _zero: ScalarT,
        /,
    ) -> None:
        (
            self._box,
            self._index,
            self._segment,
            self._segment_point_metric,
            self._segments_metric,
            self._zero,
        ) = (
            box,
            index,
            _segment,
            _segment_point_metric,
            _segments_metric,
            _zero,
        )

    __repr__ = generate_repr(__init__)

    @property
    def item(self, /) -> Item[ScalarT]:
        return self._index, self._segment

    def distance_to_point(self, point: Point[ScalarT], /) -> ScalarT:
        return (
            distance
            if (
                (distance := self._segment_point_metric(self._segment, point))
                != self._zero
            )
            else self._minus_inf()
        )

    def distance_to_segment(self, segment: Segment[ScalarT], /) -> ScalarT:
        return (
            distance
            if (
                (distance := self._segments_metric(self._segment, segment))
                != self._zero
            )
            else self._minus_inf()
        )

    @classmethod
    def _minus_inf(cls, /) -> ScalarT:
        return cast(ScalarT, -inf)


class BranchNode(HasCustomRepr, Generic[ScalarT]):
    """Represents node of segmental *R*-tree."""

    @property
    def box(self, /) -> Box[ScalarT]:
        return self._box

    @property
    def children(self, /) -> Sequence[Self | Leaf[ScalarT]]:
        return self._children

    @property
    def index(self, /) -> int:
        return self._index

    __slots__ = (
        '_box',
        '_box_point_metric',
        '_box_segment_metric',
        '_children',
        '_index',
    )

    def __init__(
        self,
        index: int,
        box: Box[ScalarT],
        children: Sequence[Self | Leaf[ScalarT]],
        _box_point_metric: _BoxPointMetric[ScalarT],
        _box_segment_metric: _BoxSegmentMetric[ScalarT],
        /,
    ) -> None:
        (
            self._box,
            self._box_point_metric,
            self._box_segment_metric,
            self._children,
            self._index,
        ) = (box, _box_point_metric, _box_segment_metric, children, index)

    __repr__ = generate_repr(__init__)

    def distance_to_point(self, point: Point[ScalarT], /) -> ScalarT:
        return self._box_point_metric(self._box, point)

    def distance_to_segment(self, segment: Segment[ScalarT], /) -> ScalarT:
        return self._box_segment_metric(self._box, segment)


AnyNode: TypeAlias = BranchNode[ScalarT] | Leaf[ScalarT]


def create_root(
    segments: Sequence[Segment[ScalarT]],
    boxes: Sequence[Box[ScalarT]],
    max_children: int,
    /,
    *,
    box_point_metric: _BoxPointMetric[ScalarT],
    box_segment_metric: _BoxSegmentMetric[ScalarT],
    boxes_merger: Callable[[Box[ScalarT], Box[ScalarT]], Box[ScalarT]],
    coordinate_factory: _CoordinateFactory[ScalarT],
    segment_point_metric: _SegmentPointMetric[ScalarT],
    segments_metric: _SegmentsMetric[ScalarT],
    zero: ScalarT,
) -> BranchNode[ScalarT]:
    leaves = [
        Leaf(index, box, segment, segment_point_metric, segments_metric, zero)
        for index, (box, segment) in enumerate(
            zip(boxes, segments, strict=True)
        )
    ]
    root_box = reduce(boxes_merger, boxes)
    leaves_count = len(leaves)
    if leaves_count <= max_children:
        # only one node, skip sorting and just fill the root box
        return BranchNode(
            len(leaves), root_box, leaves, box_point_metric, box_segment_metric
        )

    max_coordinate = coordinate_factory(hilbert.MAX_COORDINATE)
    one = coordinate_factory(1)
    two = coordinate_factory(2)

    def leaf_key(
        leaf: Leaf[ScalarT],
        /,
        *,
        double_root_delta_x: ScalarT = (
            two * (root_box.max_x - root_box.min_x) or one
        ),
        double_root_delta_y: ScalarT = (
            two * (root_box.max_y - root_box.min_y) or one
        ),
        double_root_min_x: ScalarT = two * root_box.min_x,
        double_root_min_y: ScalarT = two * root_box.min_y,
        max_coordinate: ScalarT = max_coordinate,
    ) -> int:
        box = leaf.box
        return hilbert.index(
            floor(
                (max_coordinate * (box.min_x + box.max_x - double_root_min_x))
                / double_root_delta_x
            ),
            floor(
                (max_coordinate * (box.min_y + box.max_y - double_root_min_y))
                / double_root_delta_y
            ),
        )

    leaves.sort(key=leaf_key)
    nodes: list[AnyNode[ScalarT]] = list(leaves)
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
                BranchNode(
                    len(nodes),
                    reduce(boxes_merger, [child.box for child in children]),
                    children,
                    box_point_metric,
                    box_segment_metric,
                )
            )
            start = stop
    result = nodes[-1]
    assert isinstance(result, BranchNode), result
    return result


def is_leaf(node: AnyNode[ScalarT], /) -> TypeIs[Leaf[ScalarT]]:
    return isinstance(node, Leaf)
