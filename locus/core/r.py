from functools import reduce
from math import floor
from typing import (Callable,
                    Iterator,
                    Optional,
                    Sequence,
                    Tuple)

from ground.hints import (Box,
                          Point,
                          Scalar)
from reprit.base import generate_repr

from . import hilbert
from .box import (is_subset_of,
                  overlaps)
from .utils import ceil_division

Item = Tuple[int, Box]


class Node:
    """Represents node of *R*-tree."""
    __slots__ = 'box', 'children', 'index', 'metric'

    def __init__(self,
                 index: int,
                 box: Box,
                 children: Optional[Sequence['Node']],
                 metric: Callable[[Box, Point], Scalar]) -> None:
        self.box, self.children, self.index, self.metric = (box, children,
                                                            index, metric)

    __repr__ = generate_repr(__init__)

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    @property
    def item(self) -> Item:
        return self.index, self.box

    def distance_to_point(self, point: Point) -> Scalar:
        return self.metric(self.box, point)


def create_root(boxes: Sequence[Box],
                max_children: int,
                boxes_merger: Callable[[Box, Box], Box],
                metric: Callable[[Box, Point], Scalar]) -> Node:
    nodes = [Node(index, box, None, metric) for index, box in enumerate(boxes)]
    root_box = reduce(boxes_merger, boxes)
    leaves_count = len(nodes)
    if leaves_count <= max_children:
        # only one node, skip sorting and just fill the root box
        return Node(len(nodes), root_box, nodes, metric)
    else:
        def node_key(node: Node,
                     double_root_delta_x: Scalar
                     = 2 * (root_box.max_x - root_box.min_x),
                     double_root_delta_y: Scalar
                     = 2 * (root_box.max_y - root_box.min_y),
                     double_root_min_x: Scalar = 2 * root_box.min_x,
                     double_root_min_y: Scalar = 2 * root_box.min_y) -> int:
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
                                  children, metric))
                start = stop
        return nodes[-1]


def find_node_box_subsets_items(node: Node, box: Box) -> Iterator[Item]:
    if is_subset_of(node.box, box):
        for leaf in node_to_leaves(node):
            yield leaf.item
    elif not node.is_leaf and overlaps(box, node.box):
        for child in node.children:
            yield from find_node_box_subsets_items(child, box)


def find_node_box_supersets_items(node: Node, box: Box) -> Iterator[Item]:
    if is_subset_of(box, node.box):
        if node.is_leaf:
            yield node.item
        else:
            for child in node.children:
                yield from find_node_box_supersets_items(child, box)


def node_to_leaves(node: Node) -> Iterator[Node]:
    if node.is_leaf:
        yield node
    else:
        for child in node.children:
            yield from node_to_leaves(child)
