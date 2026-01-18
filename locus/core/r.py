from collections.abc import Callable, Iterator, Sequence
from functools import reduce
from math import floor
from typing import Generic, TypeAlias

from ground.hints import Box, Point
from reprit.base import generate_repr
from typing_extensions import Self

from . import hilbert
from .box import is_subset_of, overlaps
from .hints import HasCustomRepr, ScalarT
from .utils import ceil_division

Item: TypeAlias = tuple[int, Box[ScalarT]]


class Node(HasCustomRepr, Generic[ScalarT]):
    """Represents node of *R*-tree."""

    __slots__ = 'box', 'children', 'index', 'metric'

    def __init__(
        self,
        index: int,
        box: Box[ScalarT],
        children: Sequence[Self] | None,
        metric: Callable[[Box[ScalarT], Point[ScalarT]], ScalarT],
    ) -> None:
        self.box, self.children, self.index, self.metric = (
            box,
            children,
            index,
            metric,
        )

    __repr__ = generate_repr(__init__)

    @property
    def is_leaf(self, /) -> bool:
        return self.children is None

    @property
    def item(self, /) -> Item[ScalarT]:
        return self.index, self.box

    def distance_to_point(self, point: Point[ScalarT], /) -> ScalarT:
        return self.metric(self.box, point)


def create_root(
    boxes: Sequence[Box[ScalarT]],
    max_children: int,
    boxes_merger: Callable[[Box[ScalarT], Box[ScalarT]], Box[ScalarT]],
    metric: Callable[[Box[ScalarT], Point[ScalarT]], ScalarT],
    coordinate_factory: Callable[[int], ScalarT],
) -> Node[ScalarT]:
    nodes = [Node(index, box, None, metric) for index, box in enumerate(boxes)]
    root_box = reduce(boxes_merger, boxes)
    leaves_count = len(nodes)
    if leaves_count <= max_children:
        # only one node, skip sorting and just fill the root box
        return Node(len(nodes), root_box, nodes, metric)

    two = coordinate_factory(2)

    def node_key(
        node: Node[ScalarT],
        double_root_delta_x: ScalarT = two * (root_box.max_x - root_box.min_x),
        double_root_delta_y: ScalarT = two * (root_box.max_y - root_box.min_y),
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
                    children,
                    metric,
                )
            )
            start = stop
    return nodes[-1]


def find_node_box_subsets_items(
    node: Node[ScalarT], box: Box[ScalarT], /
) -> Iterator[Item[ScalarT]]:
    if is_subset_of(node.box, box):
        for leaf in node_to_leaves(node):
            yield leaf.item
    elif not node.is_leaf and overlaps(box, node.box):
        assert node.children is not None, node
        for child in node.children:
            yield from find_node_box_subsets_items(child, box)


def find_node_box_supersets_items(
    node: Node[ScalarT], box: Box[ScalarT], /
) -> Iterator[Item[ScalarT]]:
    if is_subset_of(box, node.box):
        if node.is_leaf:
            yield node.item
        else:
            assert node.children is not None, node
            for child in node.children:
                yield from find_node_box_supersets_items(child, box)


def node_to_leaves(node: Node[ScalarT], /) -> Iterator[Node[ScalarT]]:
    if node.is_leaf:
        yield node
    else:
        assert node.children is not None, node
        for child in node.children:
            yield from node_to_leaves(child)
