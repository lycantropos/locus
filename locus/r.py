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
    def item(self) -> Item:
        return self.index, self.interval

    def distance_to_point(self, point: Point) -> Coordinate:
        return distance_to_planar_interval(point, self.interval)


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
        queue = [self._root]
        while queue:
            node = queue.pop()
            if _interval_does_not_contain(interval, node.interval):
                continue
            if node.children is None:
                yield node.item
            else:
                for child in node.children:
                    if _interval_does_not_contain(interval, child.interval):
                        continue
                    if child.children is None:
                        yield child.item
                    else:
                        queue.extend(child.children)

    def n_nearest_indices(self, n: int, point: Point) -> Sequence[int]:
        return ([index for index, _ in self.n_nearest_items(n, point)]
                if n < len(self._intervals)
                else range(len(self._intervals)))

    def n_nearest_intervals(self, n: int, point: Point) -> Sequence[Interval]:
        return ([interval for _, interval in self.n_nearest_items(n, point)]
                if n < len(self._intervals)
                else self._intervals)

    def n_nearest_items(self, n: int, point: Point) -> List[Item]:
        queue = PriorityQueue((0, 0, self._root))
        items = []
        while queue:
            node = queue.pop()[2]
            for child in node.children:
                queue.push((child.distance_to_point(point),
                            -child.index - 1
                            if child.children is None
                            else child.index,
                            child))
            while queue and queue.peek()[1] < 0:
                items.append(queue.pop()[2].item)
                if len(items) == n:
                    queue.clear()
                    return items
        queue.clear()
        return items


def _interval_does_not_contain(goal: Interval, test: Interval) -> bool:
    (goal_min_x, goal_max_x), (goal_min_y, goal_max_y) = goal
    (test_min_x, test_max_x), (test_min_y, test_max_y) = test
    return (test_max_x < goal_min_x or test_min_x > goal_max_x
            or test_max_y < goal_min_y or test_min_y > goal_max_y)


def _merge_intervals(left: Interval, right: Interval) -> Interval:
    (left_min_x, left_max_x), (left_min_y, left_max_y) = left
    (right_min_x, right_max_x), (right_min_y, right_max_y) = right
    return ((min(left_min_x, right_min_x), max(left_max_x, right_max_x)),
            (min(left_min_y, right_min_y), max(left_max_y, right_max_y)))
