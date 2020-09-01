from functools import reduce
from heapq import (heappop,
                   heappush)
from math import floor
from typing import (Iterator,
                    List,
                    Optional,
                    Sequence,
                    Tuple,
                    Type)

from reprit.base import generate_repr

from .core import (hilbert as _hilbert,
                   interval as _interval)
from .core.utils import ceil_division
from .hints import (Coordinate,
                    Interval,
                    Point)

Item = Tuple[int, Interval]


class Node:
    """
    Represents node of *R*-tree.

    Can be subclassed for custom metrics definition.
    """

    __slots__ = 'index', 'interval', 'children'

    def __init__(self,
                 index: int,
                 interval: Interval,
                 children: Optional[Sequence['Node']]) -> None:
        """
        Initializes node.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> node = Node(5, ((-10, 10), (0, 20)), None)
        """
        self.index = index
        self.interval = interval
        self.children = children

    __repr__ = generate_repr(__init__)

    @property
    def is_leaf(self) -> bool:
        """
        Checks whether the node is a leaf.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> node = Node(5, ((-10, 10), (0, 20)), None)
        >>> node.is_leaf
        True
        """
        return self.children is None

    @property
    def item(self) -> Item:
        """
        Returns underlying index with interval.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> node = Node(5, ((-10, 10), (0, 20)), None)
        >>> node.item == (5, ((-10, 10), (0, 20)))
        True
        """
        return self.index, self.interval

    def distance_to_point(self, point: Point) -> Coordinate:
        """
        Calculates distance to given point.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> node = Node(5, ((-10, 10), (0, 20)), None)
        >>> node.distance_to_point((20, 0)) == 10
        True
        """
        return _interval.planar_distance_to_point(self.interval, point)


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

    def find_subsets(self, interval: Interval) -> List[Interval]:
        """
        Searches for intervals that lie inside the given interval.

        Time complexity:
            ``O(max_children * log size + hits_count)``
        Memory complexity:
            ``O(max_children * log size + hits_count)``

        where ``size = len(self.intervals)``,
        ``max_children = self.max_children``,
        ``hits_count`` --- number of found intervals.

        :param interval: input interval.
        :returns: intervals that lie inside the input interval.

        >>> intervals = [((-index, index), (0, index))
        ...              for index in range(1, 11)]
        >>> tree = Tree(intervals)
        >>> tree.find_subsets(((-1, 1), (0, 1))) == [((-1, 1), (0, 1))]
        True
        >>> (tree.find_subsets(((-2, 2), (0, 2)))
        ...  == [((-1, 1), (0, 1)), ((-2, 2), (0, 2))])
        True
        >>> (tree.find_subsets(((-3, 3), (0, 3)))
        ...  == [((-1, 1), (0, 1)), ((-2, 2), (0, 2)), ((-3, 3), (0, 3))])
        True
        """
        return [interval
                for _, interval in self._find_subsets_items(interval)]

    def find_subsets_indices(self, interval: Interval) -> List[int]:
        """
        Searches for indices of intervals that lie inside the given interval.

        Time complexity:
            ``O(max_children * log size + hits_count)``
        Memory complexity:
            ``O(max_children * log size + hits_count)``

        where ``size = len(self.intervals)``,
        ``max_children = self.max_children``,
        ``hits_count`` --- number of found indices.

        :param interval: input interval.
        :returns: indices of intervals that lie inside the input interval.

        >>> intervals = [((-index, index), (0, index))
        ...              for index in range(1, 11)]
        >>> tree = Tree(intervals)
        >>> tree.find_subsets_indices(((-1, 1), (0, 1))) == [0]
        True
        >>> tree.find_subsets_indices(((-2, 2), (0, 2))) == [0, 1]
        True
        >>> tree.find_subsets_indices(((-3, 3), (0, 3))) == [0, 1, 2]
        True
        """
        return [index for index, _ in self._find_subsets_items(interval)]

    def find_subsets_items(self, interval: Interval) -> List[Item]:
        """
        Searches for indices with intervals that lie inside the given interval.

        Time complexity:
            ``O(max_children * log size + hits_count)``
        Memory complexity:
            ``O(max_children * log size + hits_count)``

        where ``size = len(self.intervals)``,
        ``max_children = self.max_children``,
        ``hits_count`` --- number of found indices with intervals.

        :param interval: input interval.
        :returns: indices with intervals that lie inside the input interval.

        >>> intervals = [((-index, index), (0, index))
        ...              for index in range(1, 11)]
        >>> tree = Tree(intervals)
        >>> (tree.find_subsets_items(((-1, 1), (0, 1)))
        ...  == [(0, ((-1, 1), (0, 1)))])
        True
        >>> (tree.find_subsets_items(((-2, 2), (0, 2)))
        ...  == [(0, ((-1, 1), (0, 1))), (1, ((-2, 2), (0, 2)))])
        True
        >>> (tree.find_subsets_items(((-3, 3), (0, 3)))
        ...  == [(0, ((-1, 1), (0, 1))), (1, ((-2, 2), (0, 2))),
        ...      (2, ((-3, 3), (0, 3)))])
        True
        """
        return list(self._find_subsets_items(interval))

    def find_supersets(self, interval: Interval) -> List[Interval]:
        """
        Searches for intervals that contain the given interval.

        Time complexity:
            ``O(max_children * log size + hits_count)``
        Memory complexity:
            ``O(max_children * log size + hits_count)``

        where ``size = len(self.intervals)``,
        ``max_children = self.max_children``,
        ``hits_count`` --- number of found intervals.

        :param interval: input interval.
        :returns: intervals that contain the input interval.

        >>> intervals = [((-index, index), (0, index))
        ...              for index in range(1, 11)]
        >>> tree = Tree(intervals)
        >>> tree.find_supersets(((-10, 10), (0, 10))) == [((-10, 10), (0, 10))]
        True
        >>> (tree.find_supersets(((-9, 9), (0, 9)))
        ...  == [((-9, 9), (0, 9)), ((-10, 10), (0, 10))])
        True
        >>> (tree.find_supersets(((-8, 8), (0, 8)))
        ...  == [((-8, 8), (0, 8)), ((-9, 9), (0, 9)), ((-10, 10), (0, 10))])
        True
        """
        return [interval
                for _, interval in self._find_supersets_items(interval)]

    def find_supersets_indices(self, interval: Interval) -> List[int]:
        """
        Searches for indices of intervals that contain the given interval.

        Time complexity:
            ``O(max_children * log size + hits_count)``
        Memory complexity:
            ``O(max_children * log size + hits_count)``

        where ``size = len(self.intervals)``,
        ``max_children = self.max_children``,
        ``hits_count`` --- number of found indices.

        :param interval: input interval.
        :returns: indices of intervals that contain the input interval.

        >>> intervals = [((-index, index), (0, index))
        ...              for index in range(1, 11)]
        >>> tree = Tree(intervals)
        >>> tree.find_supersets_indices(((-10, 10), (0, 10))) == [9]
        True
        >>> tree.find_supersets_indices(((-9, 9), (0, 9))) == [8, 9]
        True
        >>> tree.find_supersets_indices(((-8, 8), (0, 8))) == [7, 8, 9]
        True
        """
        return [index for index, _ in self._find_supersets_items(interval)]

    def find_supersets_items(self, interval: Interval) -> List[Item]:
        """
        Searches for indices with intervals
        that contain the given interval.

        Time complexity:
            ``O(max_children * log size + hits_count)``
        Memory complexity:
            ``O(max_children * log size + hits_count)``

        where ``size = len(self.intervals)``,
        ``max_children = self.max_children``,
        ``hits_count`` --- number of found indices with intervals.

        :param interval: input interval.
        :returns: indices with intervals that contain the input interval.

        >>> intervals = [((-index, index), (0, index))
        ...              for index in range(1, 11)]
        >>> tree = Tree(intervals)
        >>> (tree.find_supersets_items(((-10, 10), (0, 10)))
        ...  == [(9, ((-10, 10), (0, 10)))])
        True
        >>> (tree.find_supersets_items(((-9, 9), (0, 9)))
        ...  == [(8, ((-9, 9), (0, 9))), (9, ((-10, 10), (0, 10)))])
        True
        >>> (tree.find_supersets_items(((-8, 8), (0, 8)))
        ...  == [(7, ((-8, 8), (0, 8))), (8, ((-9, 9), (0, 9))),
        ...      (9, ((-10, 10), (0, 10)))])
        True
        """
        return list(self._find_supersets_items(interval))

    def _find_subsets_items(self, interval: Interval) -> Iterator[Item]:
        yield from (enumerate(self._intervals)
                    if _interval.is_subset_of(self._root.interval, interval)
                    else _find_node_interval_subsets_items(self._root,
                                                           interval))

    def _find_supersets_items(self, interval: Interval) -> Iterator[Item]:
        yield from _find_node_interval_supersets_items(self._root, interval)

    def n_nearest_indices(self, n: int, point: Point) -> Sequence[int]:
        """
        Searches for indices of intervals in the tree
        the nearest to the given point.

        Time complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(1)`` otherwise
        Memory complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(1)`` otherwise

        where ``size = len(self.intervals)``,
        ``max_children = self.max_children``.

        :param n: positive upper bound for number of result indices.
        :param point: input point.
        :returns:
            indices of intervals in the tree the nearest to the input point.

        >>> intervals = [((-index, index), (0, index))
        ...              for index in range(1, 11)]
        >>> tree = Tree(intervals)
        >>> tree.n_nearest_indices(2, (0, 0)) == [9, 8]
        True
        >>> (tree.n_nearest_indices(len(intervals), (0, 0))
        ...  == range(len(intervals)))
        True
        """
        return ([index for index, _ in self._n_nearest_items(n, point)]
                if n < len(self._intervals)
                else range(len(self._intervals)))

    def n_nearest_intervals(self, n: int, point: Point) -> Sequence[Interval]:
        """
        Searches for intervals in the tree the nearest to the given point.

        Time complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(1)`` otherwise
        Memory complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(1)`` otherwise

        where ``size = len(self.intervals)``,
        ``max_children = self.max_children``.

        :param n: positive upper bound for number of result intervals.
        :param point: input point.
        :returns: intervals in the tree the nearest to the input point.

        >>> intervals = [((-index, index), (0, index))
        ...              for index in range(1, 11)]
        >>> tree = Tree(intervals)
        >>> (tree.n_nearest_intervals(2, (0, 0))
        ...  == [((-10, 10), (0, 10)), ((-9, 9), (0, 9))])
        True
        >>> tree.n_nearest_intervals(len(intervals), (0, 0)) == intervals
        True
        """
        return ([interval for _, interval in self._n_nearest_items(n, point)]
                if n < len(self._intervals)
                else self._intervals)

    def n_nearest_items(self, n: int, point: Point) -> Sequence[Item]:
        """
        Searches for indices with intervals in the tree
        the nearest to the given point.

        Time complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(size)`` otherwise
        Memory complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(size)`` otherwise

        where ``size = len(self.intervals)``,
        ``max_children = self.max_children``.

        :param n:
            positive upper bound for number of result indices with intervals.
        :param point: input point.
        :returns:
            indices with intervals in the tree the nearest to the input point.

        >>> intervals = [((-index, index), (0, index))
        ...              for index in range(1, 11)]
        >>> tree = Tree(intervals)
        >>> (tree.n_nearest_items(2, (0, 0))
        ...  == [(9, ((-10, 10), (0, 10))), (8, ((-9, 9), (0, 9)))])
        True
        >>> (tree.n_nearest_items(len(intervals), (0, 0))
        ...  == list(enumerate(intervals)))
        True
        """
        return list(self._n_nearest_items(n, point)
                    if n < len(self._intervals)
                    else enumerate(self._intervals))

    def _n_nearest_items(self, n: int, point: Point) -> Iterator[Item]:
        queue = [(0, 0, self._root)]
        while n and queue:
            _, _, node = heappop(queue)
            for child in node.children:
                heappush(queue,
                         (child.distance_to_point(point),
                          -child.index - 1 if child.is_leaf else child.index,
                          child))
            while n and queue and queue[0][1] < 0:
                _, _, node = heappop(queue)
                yield node.item
                n -= 1

    def nearest_index(self, point: Point) -> int:
        """
        Searches for index of interval in the tree
        the nearest to the given point.

        Time complexity:
            ``O(max_children * log size)``
        Memory complexity:
            ``O(max_children * log size)``

        where ``size = len(self.intervals)``,
        ``max_children = self.max_children``.

        :param point: input point.
        :returns: index of interval in the tree the nearest to the input point.

        >>> intervals = [((-index, index), (0, index))
        ...              for index in range(1, 11)]
        >>> tree = Tree(intervals)
        >>> tree.nearest_index((0, 0)) == 9
        True
        """
        result, _ = self.nearest_item(point)
        return result

    def nearest_interval(self, point: Point) -> Interval:
        """
        Searches for interval in the tree the nearest to the given point.

        Time complexity:
            ``O(max_children * log size)``
        Memory complexity:
            ``O(max_children * log size)``

        where ``size = len(self.intervals)``,
        ``max_children = self.max_children``.

        :param point: input point.
        :returns: interval in the tree the nearest to the input point.

        >>> intervals = [((-index, index), (0, index))
        ...              for index in range(1, 11)]
        >>> tree = Tree(intervals)
        >>> tree.nearest_interval((0, 0)) == ((-10, 10), (0, 10))
        True
        """
        _, result = self.nearest_item(point)
        return result

    def nearest_item(self, point: Point) -> Item:
        """
        Searches for index with interval in the tree
        the nearest to the given point.

        Time complexity:
            ``O(max_children * log size)``
        Memory complexity:
            ``O(max_children * log size)``

        where ``size = len(self.intervals)``,
        ``max_children = self.max_children``.

        :param point: input point.
        :returns:
            index with interval in the tree the nearest to the input point.

        >>> intervals = [((-index, index), (0, index))
        ...              for index in range(1, 11)]
        >>> tree = Tree(intervals)
        >>> tree.nearest_item((0, 0)) == (9, ((-10, 10), (0, 10)))
        True
        >>> tree.nearest_item((-10, 0)) == (9, ((-10, 10), (0, 10)))
        True
        >>> tree.nearest_item((-10, 10)) == (9, ((-10, 10), (0, 10)))
        True
        >>> tree.nearest_item((10, 0)) == (9, ((-10, 10), (0, 10)))
        True
        >>> tree.nearest_item((10, 10)) == (9, ((-10, 10), (0, 10)))
        True
        """
        queue = [(0, 0, self._root)]
        while queue:
            _, _, node = heappop(queue)
            for child in node.children:
                heappush(queue,
                         (child.distance_to_point(point),
                          -child.index - 1 if child.is_leaf else child.index,
                          child))
            if queue and queue[0][1] < 0:
                _, _, node = heappop(queue)
                return node.item


def _create_root(intervals: Sequence[Interval],
                 max_children: int,
                 node_cls: Type[Node] = Node) -> Node:
    intervals_count = len(intervals)
    nodes = [node_cls(index, interval, None)
             for index, interval in enumerate(intervals)]
    interval = reduce(_interval.merge, intervals)
    if intervals_count <= max_children:
        # only one node, skip sorting and just fill the root box
        return node_cls(len(nodes), interval, nodes)
    else:
        (tree_min_x, tree_max_x), (tree_min_y, tree_max_y) = interval

        def node_key(node: Node,
                     double_tree_delta_x: Coordinate
                     = 2 * (tree_max_x - tree_min_x),
                     double_tree_delta_y: Coordinate
                     = 2 * (tree_max_y - tree_min_y),
                     double_tree_min_x: Coordinate = 2 * tree_min_x,
                     double_tree_min_y: Coordinate = 2 * tree_min_y) -> int:
            (min_x, max_x), (min_y, max_y) = node.interval
            return _hilbert.index(floor(_hilbert.MAX_COORDINATE
                                        * (min_x + max_x - double_tree_min_x)
                                        / double_tree_delta_x),
                                  floor(_hilbert.MAX_COORDINATE
                                        * (min_y + max_y - double_tree_min_y)
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
                                      reduce(_interval.merge,
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


def _find_node_interval_subsets_items(node: Node,
                                      interval: Interval) -> Iterator[Item]:
    if _interval.is_subset_of(node.interval, interval):
        for leaf in _node_to_leaves(node):
            yield leaf.item
    elif not node.is_leaf and _interval.overlaps(interval, node.interval):
        for child in node.children:
            yield from _find_node_interval_subsets_items(child, interval)


def _find_node_interval_supersets_items(node: Node,
                                        interval: Interval) -> Iterator[Item]:
    if _interval.is_subset_of(interval, node.interval):
        if node.is_leaf:
            yield node.item
        else:
            for child in node.children:
                yield from _find_node_interval_supersets_items(child, interval)
