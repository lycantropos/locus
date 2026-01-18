from collections.abc import Callable, Iterator, Sequence
from functools import reduce
from math import floor
from typing import Generic, TypeAlias

from ground.hints import Box, Point
from reprit.base import generate_repr
from typing_extensions import Self, TypeIs

from . import hilbert
from .box import is_subset_of, overlaps
from .hints import HasCustomRepr, ScalarT
from .utils import ceil_division

Item: TypeAlias = tuple[int, Box[ScalarT]]
_BoxPointMetric: TypeAlias = Callable[[Box[ScalarT], Point[ScalarT]], ScalarT]


class Leaf(HasCustomRepr, Generic[ScalarT]):
    """Represents leaf of *R*-tree."""

    @property
    def box(self, /) -> Box[ScalarT]:
        return self._box

    @property
    def index(self, /) -> int:
        return self._index

    @property
    def item(self, /) -> Item[ScalarT]:
        return self._index, self._box

    def distance_to_point(self, point: Point[ScalarT], /) -> ScalarT:
        return self._metric(self._box, point)

    __slots__ = '_box', '_index', '_metric'

    def __init__(
        self,
        index: int,
        box: Box[ScalarT],
        _metric: _BoxPointMetric[ScalarT],
        /,
    ) -> None:
        self._box, self._index, self._metric = box, index, _metric

    __repr__ = generate_repr(__init__)


class BranchNode(HasCustomRepr, Generic[ScalarT]):
    """Represents branch node of *R*-tree."""

    @property
    def box(self, /) -> Box[ScalarT]:
        return self._box

    @property
    def children(self, /) -> Sequence[Self | Leaf[ScalarT]]:
        return self._children

    @property
    def index(self, /) -> int:
        return self._index

    def distance_to_point(self, point: Point[ScalarT], /) -> ScalarT:
        return self._metric(self._box, point)

    __slots__ = '_box', '_children', '_index', '_metric'

    def __init__(
        self,
        index: int,
        box: Box[ScalarT],
        _metric: _BoxPointMetric[ScalarT],
        /,
        *,
        children: Sequence[Self | Leaf[ScalarT]],
    ) -> None:
        self._box, self._children, self._index, self._metric = (
            box,
            children,
            index,
            _metric,
        )

    __repr__ = generate_repr(__init__)


AnyNode: TypeAlias = BranchNode[ScalarT] | Leaf[ScalarT]


def create_root(
    boxes: Sequence[Box[ScalarT]],
    max_children: int,
    /,
    *,
    boxes_merger: Callable[[Box[ScalarT], Box[ScalarT]], Box[ScalarT]],
    coordinate_factory: Callable[[int], ScalarT],
    metric: _BoxPointMetric[ScalarT],
) -> BranchNode[ScalarT]:
    leaves = [Leaf(index, box, metric) for index, box in enumerate(boxes)]
    root_box = reduce(boxes_merger, boxes)
    leaves_count = len(leaves)
    if leaves_count <= max_children:
        # only one node, skip sorting and just fill the root box
        return BranchNode(len(leaves), root_box, metric, children=leaves)

    two = coordinate_factory(2)
    max_coordinate = coordinate_factory(hilbert.MAX_COORDINATE)

    def leaf_key(
        node: Leaf[ScalarT],
        /,
        *,
        double_root_delta_x: ScalarT = two * (root_box.max_x - root_box.min_x),
        double_root_delta_y: ScalarT = two * (root_box.max_y - root_box.min_y),
        double_root_min_x: ScalarT = two * root_box.min_x,
        double_root_min_y: ScalarT = two * root_box.min_y,
        max_coordinate: ScalarT = max_coordinate,
    ) -> int:
        box = node.box
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
    del leaves
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
                    metric,
                    children=children,
                )
            )
            start = stop
    result = nodes[-1]
    assert isinstance(result, BranchNode), result
    return result


def find_node_box_subsets_items(
    node: AnyNode[ScalarT], box: Box[ScalarT], /
) -> Iterator[Item[ScalarT]]:
    if is_subset_of(node.box, box):
        for leaf in node_to_leaves(node):
            yield leaf.item
    elif isinstance(node, BranchNode) and overlaps(box, node.box):
        for child in node.children:
            yield from find_node_box_subsets_items(child, box)


def find_node_box_supersets_items(
    node: AnyNode[ScalarT], box: Box[ScalarT], /
) -> Iterator[Item[ScalarT]]:
    if is_subset_of(box, node.box):
        if is_leaf(node):
            yield node.item
        else:
            for child in node.children:
                yield from find_node_box_supersets_items(child, box)


def is_leaf(node: AnyNode[ScalarT], /) -> TypeIs[Leaf[ScalarT]]:
    return isinstance(node, Leaf)


def node_to_leaves(node: AnyNode[ScalarT], /) -> Iterator[Leaf[ScalarT]]:
    if is_leaf(node):
        yield node
    else:
        for child in node.children:
            yield from node_to_leaves(child)
