from math import floor
from functools import reduce
from typing import (Iterator,
                    List,
                    Optional,
                    Sequence,
                    Tuple,
                    Type)

from prioq.base import PriorityQueue
from reprit.base import generate_repr

from .core.utils import (HILBERT_MAX_COORDINATE,
                         ceil_division,
                         distance_to_planar_interval,
                         to_hilbert_index)
from .hints import (Coordinate,
                    Interval,
                    Point)

Item = Tuple[int, Interval]


class Node:
    __slots__ = 'index', 'interval', 'children'

    def __init__(self,
                 index: int,
                 interval: Interval,
                 children: Optional[Sequence['Node']]) -> None:
        self.index = index
        self.interval = interval
        self.children = children

    __repr__ = generate_repr(__init__)

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    @property
    def item(self) -> Item:
        return self.index, self.interval

    def distance_to_point(self, point: Point) -> Coordinate:
        return distance_to_planar_interval(point, self.interval)


class Tree:
    """
    Represents packed 2-dimensional Hilbert *R*-tree.

    Reference:
        https://en.wikipedia.org/wiki/Hilbert_R-tree#Packed_Hilbert_R-trees
    """

    def __init__(self,
                 intervals: Sequence[Interval],
                 *,
                 max_children: int = 16,
                 node_cls: Type[Node] = Node) -> None:
        """
        Initializes tree from intervals.


        Time complexity:
            ``O(size * log size)``
        Memory complexity:
            ``O(size)``

        where ``size = len(intervals)``.

        >>> intervals = [((-index, index), (0, index))
        ...              for index in range(1, 11)]
        >>> tree = Tree(intervals)
        """
        self._intervals = intervals
        self._max_children = max_children
        self._root = _create_root(intervals, max_children, node_cls)

    __repr__ = generate_repr(__init__)

    @property
    def intervals(self) -> Sequence[Interval]:
        """
        Returns underlying intervals.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> intervals = [((-index, index), (0, index))
        ...              for index in range(1, 11)]
        >>> tree = Tree(intervals)
        >>> tree.intervals == intervals
        True
        """
        return self._intervals

    @property
    def node_cls(self) -> Type[Node]:
        """
        Returns type of the nodes.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> intervals = [((-index, index), (0, index))
        ...              for index in range(1, 11)]
        >>> tree = Tree(intervals)
        >>> tree.node_cls is Node
        True
        """
        return type(self._root)

    @property
    def max_children(self) -> int:
        """
        Returns maximum number of children in each node.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> intervals = [((-index, index), (0, index))
        ...              for index in range(1, 11)]
        >>> tree = Tree(intervals)
        >>> tree.max_children == 16
        True
        """
        return self._max_children

    def find_interval_indices(self, interval: Interval) -> List[int]:
        return [index for index, _ in self._find_interval_items(interval)]

    def find_interval_intervals(self, interval: Interval) -> List[Interval]:
        return [interval
                for _, interval in self._find_interval_items(interval)]

    def find_interval_items(self, interval: Interval) -> List[Item]:
        return list(self._find_interval_items(interval))

    def _find_interval_items(self, interval: Interval) -> Iterator[Item]:
        yield from (enumerate(self._intervals)
                    if _is_interval_subset_of(self._root.interval, interval)
                    else _find_node_interval_items(self._root, interval))

    def n_nearest_indices(self, n: int, point: Point) -> Sequence[int]:
        return ([index for index, _ in self._n_nearest_items(n, point)]
                if n < len(self._intervals)
                else range(len(self._intervals)))

    def n_nearest_intervals(self, n: int, point: Point) -> Sequence[Interval]:
        return ([interval for _, interval in self._n_nearest_items(n, point)]
                if n < len(self._intervals)
                else self._intervals)

    def n_nearest_items(self, n: int, point: Point) -> Sequence[Item]:
        return (self._n_nearest_items(n, point)
                if n < len(self._intervals)
                else list(enumerate(self._intervals)))

    def _n_nearest_items(self, n: int, point: Point) -> List[Item]:
        queue = PriorityQueue((0, 0, self._root))
        items = []
        while queue:
            node = queue.pop()[2]
            for child in node.children:
                queue.push((child.distance_to_point(point),
                            -child.index - 1 if child.is_leaf else child.index,
                            child))
            while queue and queue.peek()[1] < 0:
                items.append(queue.pop()[2].item)
                if len(items) == n:
                    return items
        return items


def _create_root(intervals: Sequence[Interval],
                 max_children: int,
                 node_cls: Type[Node] = Node) -> Node:
    intervals_count = len(intervals)
    nodes = [node_cls(index, interval, None)
             for index, interval in enumerate(intervals)]
    interval = reduce(_merge_intervals, intervals)
    if intervals_count <= max_children:
        # only one node, skip sorting and just fill the root box
        return node_cls(len(nodes), interval, nodes)
    else:
        (x_min, x_max), (y_min, y_max) = interval

        def node_key(node: Node,
                     double_tree_delta_x: Coordinate = 2 * (x_max - x_min),
                     double_tree_delta_y: Coordinate = 2 * (y_max - y_min),
                     double_tree_x_min: Coordinate = 2 * x_min,
                     double_tree_y_min: Coordinate = 2 * y_min) -> int:
            (x_min, x_max), (y_min, y_max) = node.interval
            return to_hilbert_index(floor(HILBERT_MAX_COORDINATE
                                          * (x_min + x_max - double_tree_x_min)
                                          / double_tree_delta_x),
                                    floor(HILBERT_MAX_COORDINATE
                                          * (y_min + y_max - double_tree_y_min)
                                          / double_tree_delta_y))

        nodes = sorted(nodes,
                       key=node_key)
        nodes_count = step = len(intervals)
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
                nodes.append(node_cls(len(nodes),
                                      reduce(_merge_intervals,
                                             [child.interval
                                              for child in children]),
                                      children))
                start = stop
        return nodes[-1]


def _node_to_leaves(node: Node) -> Iterator[Node]:
    if node.is_leaf:
        yield node
    elif node.children[0].is_leaf:
        yield from node.children
    else:
        for child in node.children:
            yield from _node_to_leaves(child)


def _find_node_interval_items(node: Node,
                              interval: Interval) -> Iterator[Item]:
    if _is_interval_subset_of(node.interval, interval):
        for node_leaf in _node_to_leaves(node):
            yield node_leaf.item
    elif (not node.is_leaf
          and _is_interval_subset_of(interval, node.interval)):
        for child in node.children:
            yield from _find_node_interval_items(child, interval)


def _is_interval_subset_of(test: Interval, goal: Interval) -> bool:
    (goal_x_min, goal_x_max), (goal_y_min, goal_y_max) = goal
    (test_x_min, test_x_max), (test_y_min, test_y_max) = test
    return (goal_x_min <= test_x_min and test_x_max <= goal_x_max
            and goal_y_min <= test_y_min and test_y_max <= goal_y_max)


def _merge_intervals(left: Interval, right: Interval) -> Interval:
    (left_x_min, left_x_max), (left_y_min, left_y_max) = left
    (right_x_min, right_x_max), (right_y_min, right_y_max) = right
    return ((min(left_x_min, right_x_min), max(left_x_max, right_x_max)),
            (min(left_y_min, right_y_min), max(left_y_max, right_y_max)))
