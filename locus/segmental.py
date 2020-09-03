from functools import reduce
from heapq import (heappop,
                   heappush)
from math import (floor,
                  inf)
from typing import (Iterator,
                    Optional,
                    Sequence,
                    Tuple,
                    Type)

from reprit.base import generate_repr

from .core import (hilbert as _hilbert,
                   interval as _interval,
                   segment as _segment)
from .core.utils import ceil_division
from .hints import (Coordinate,
                    Interval,
                    Point,
                    Segment)

Item = Tuple[int, Segment]


class Node:
    """
    Represents node of segmental *R*-tree.

    Can be subclassed for custom metrics definition.
    """

    __slots__ = 'index', 'interval', 'segment', 'children'

    def __init__(self,
                 index: int,
                 interval: Interval,
                 segment: Optional[Segment],
                 children: Optional[Sequence['Node']]) -> None:
        """
        Initializes node.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> node = Node(5, ((-10, 10), (0, 20)), ((-10, 20), (10, 0)), None)
        """
        self.index = index
        self.interval = interval
        self.segment = segment
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

        >>> node = Node(5, ((-10, 10), (0, 20)), ((-10, 20), (10, 0)), None)
        >>> node.is_leaf
        True
        """
        return self.children is None

    @property
    def item(self) -> Item:
        """
        Returns underlying index with segment.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> node = Node(5, ((-10, 10), (0, 20)), ((-10, 20), (10, 0)), None)
        >>> node.item == (5, ((-10, 20), (10, 0)))
        True
        """
        return self.index, self.segment

    def distance_to_point(self, point: Point,
                          *,
                          _minus_inf: Coordinate = -inf) -> Coordinate:
        """
        Calculates distance to given point.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> node = Node(5, ((-10, 10), (0, 20)), ((-10, 20), (10, 0)), None)
        >>> node.distance_to_point((20, 0)) == 10
        True
        """
        return (_segment.distance_to_point(self.segment, point) or _minus_inf
                if self.is_leaf
                else _interval.planar_distance_to_point(self.interval, point))

    def distance_to_segment(self, segment: Segment,
                            *,
                            _minus_inf: Coordinate = -inf) -> Coordinate:
        """
        Calculates distance to given segment.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> node = Node(5, ((-10, 10), (0, 20)), ((-10, 20), (10, 0)), None)
        >>> node.distance_to_segment(((20, 0), (20, 10))) == 10
        True
        """
        return (_segment.distance_to(self.segment, segment) or _minus_inf
                if self.is_leaf
                else _segment.distance_to_interval(segment, self.interval))


class Tree:
    """
    Represents packed 2-dimensional segmental Hilbert *R*-tree.

    Reference:
        https://en.wikipedia.org/wiki/Hilbert_R-tree#Packed_Hilbert_R-trees
    """
    __slots__ = '_segments', '_max_children', '_root'

    def __init__(self,
                 segments: Sequence[Segment],
                 *,
                 max_children: int = 16,
                 node_cls: Type[Node] = Node) -> None:
        """
        Initializes tree from segments.


        Time complexity:
            ``O(size * log size)``
        Memory complexity:
            ``O(size)``

        where ``size = len(segments)``.

        >>> segments = [((0, index), (index, index)) for index in range(1, 11)]
        >>> tree = Tree(segments)
        """
        self._segments = segments
        self._max_children = max_children
        self._root = _create_root(segments, max_children, node_cls)

    __repr__ = generate_repr(__init__)

    @property
    def segments(self) -> Sequence[Segment]:
        """
        Returns underlying segments.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> segments = [((0, index), (index, index)) for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> tree.segments == segments
        True
        """
        return self._segments

    @property
    def node_cls(self) -> Type[Node]:
        """
        Returns type of the nodes.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> segments = [((0, index), (index, index)) for index in range(1, 11)]
        >>> tree = Tree(segments)
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

        >>> segments = [((0, index), (index, index)) for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> tree.max_children == 16
        True
        """
        return self._max_children

    def n_nearest_indices(self, n: int, segment: Segment) -> Sequence[int]:
        """
        Searches for indices of segments in the tree
        the nearest to the given segment.

        Time complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(1)`` otherwise
        Memory complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(1)`` otherwise

        where ``size = len(self.segments)``,
        ``max_children = self.max_children``.

        :param n: positive upper bound for number of result indices.
        :param segment: input segment.
        :returns:
            indices of segments in the tree the nearest to the input segment.

        >>> segments = [((0, index), (index, index)) for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> tree.n_nearest_indices(2, ((0, 0), (10, 0))) == [0, 1]
        True
        >>> (tree.n_nearest_indices(10, ((0, 0), (10, 0)))
        ...  == range(len(segments)))
        True
        """
        return ([index for index, _ in self._n_nearest_items(n, segment)]
                if n < len(self._segments)
                else range(len(self._segments)))

    def n_nearest_items(self, n: int, segment: Segment) -> Sequence[Item]:
        """
        Searches for indices with segments in the tree
        the nearest to the given segment.

        Time complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(size)`` otherwise
        Memory complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(size)`` otherwise

        where ``size = len(self.segments)``,
        ``max_children = self.max_children``.

        :param n:
            positive upper bound for number of result indices with segments.
        :param segment: input segment.
        :returns:
            indices with segments in the tree the nearest to the input segment.

        >>> segments = [((0, index), (index, index)) for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> (tree.n_nearest_items(2, ((0, 0), (10, 0)))
        ...  == [(0, ((0, 1), (1, 1))), (1, ((0, 2), (2, 2)))])
        True
        >>> (tree.n_nearest_items(10, ((0, 0), (10, 0)))
        ...  == list(enumerate(segments)))
        True
        """
        return list(self._n_nearest_items(n, segment)
                    if n < len(self._segments)
                    else enumerate(self._segments))

    def n_nearest_segments(self, n: int, segment: Segment
                           ) -> Sequence[Segment]:
        """
        Searches for segments in the tree the nearest to the given segment.

        Time complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(1)`` otherwise
        Memory complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(1)`` otherwise

        where ``size = len(self.segments)``,
        ``max_children = self.max_children``.

        :param n: positive upper bound for number of result segments.
        :param segment: input segment.
        :returns: segments in the tree the nearest to the input segment.

        >>> segments = [((0, index), (index, index)) for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> (tree.n_nearest_segments(2, ((0, 0), (10, 0)))
        ...  == [((0, 1), (1, 1)), ((0, 2), (2, 2))])
        True
        >>> tree.n_nearest_segments(10, ((0, 0), (10, 0))) == segments
        True
        """
        return ([segment for _, segment in self._n_nearest_items(n, segment)]
                if n < len(self._segments)
                else self._segments)

    def n_nearest_to_point_indices(self, n: int, point: Point
                                   ) -> Sequence[int]:
        """
        Searches for indices of segments in the tree
        the nearest to the given point.

        Time complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(1)`` otherwise
        Memory complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(1)`` otherwise

        where ``size = len(self.segments)``,
        ``max_children = self.max_children``.

        :param n: positive upper bound for number of result indices.
        :param point: input point.
        :returns:
            indices of segments in the tree the nearest to the input point.

        >>> segments = [((0, index), (index, index)) for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> tree.n_nearest_to_point_indices(2, (0, 0)) == [0, 1]
        True
        >>> tree.n_nearest_to_point_indices(10, (0, 0)) == range(len(segments))
        True
        """
        return ([index
                 for index, _ in self._n_nearest_to_point_items(n, point)]
                if n < len(self._segments)
                else range(len(self._segments)))

    def n_nearest_to_point_items(self, n: int, point: Point) -> Sequence[Item]:
        """
        Searches for indices with segments in the tree
        the nearest to the given point.

        Time complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(size)`` otherwise
        Memory complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(size)`` otherwise

        where ``size = len(self.segments)``,
        ``max_children = self.max_children``.

        :param n:
            positive upper bound for number of result indices with segments.
        :param point: input point.
        :returns:
            indices with segments in the tree the nearest to the input point.

        >>> segments = [((0, index), (index, index)) for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> (tree.n_nearest_to_point_items(2, (0, 0))
        ...  == [(0, ((0, 1), (1, 1))), (1, ((0, 2), (2, 2)))])
        True
        >>> (tree.n_nearest_to_point_items(10, (0, 0))
        ...  == list(enumerate(segments)))
        True
        """
        return list(self._n_nearest_to_point_items(n, point)
                    if n < len(self._segments)
                    else enumerate(self._segments))

    def n_nearest_to_point_segments(self, n: int, point: Point
                                    ) -> Sequence[Segment]:
        """
        Searches for segments in the tree the nearest to the given point.

        Time complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(1)`` otherwise
        Memory complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(1)`` otherwise

        where ``size = len(self.segments)``,
        ``max_children = self.max_children``.

        :param n: positive upper bound for number of result segments.
        :param point: input point.
        :returns: segments in the tree the nearest to the input point.

        >>> segments = [((0, index), (index, index)) for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> (tree.n_nearest_to_point_segments(2, (0, 0))
        ...  == [((0, 1), (1, 1)), ((0, 2), (2, 2))])
        True
        >>> tree.n_nearest_to_point_segments(10, (0, 0)) == segments
        True
        """
        return ([segment
                 for _, segment in self._n_nearest_to_point_items(n, point)]
                if n < len(self._segments)
                else self._segments)

    def nearest_index(self, segment: Segment) -> int:
        """
        Searches for index of segment in the tree
        the nearest to the given segment.

        Time complexity:
            ``O(max_children * log size)``
        Memory complexity:
            ``O(max_children * log size)``

        where ``size = len(self.segments)``,
        ``max_children = self.max_children``.

        :param segment: input segment.
        :returns:
            index of segment in the tree the nearest to the input segment.

        >>> segments = [((0, index), (index, index)) for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> tree.nearest_index(((0, 0), (10, 0))) == 0
        True
        """
        result, _ = self.nearest_item(segment)
        return result

    def nearest_item(self, segment: Segment) -> Item:
        """
        Searches for index with segment in the tree
        the nearest to the given segment.

        Time complexity:
            ``O(max_children * log size)``
        Memory complexity:
            ``O(max_children * log size)``

        where ``size = len(self.segments)``,
        ``max_children = self.max_children``.

        :param segment: input segment.
        :returns:
            index with segment in the tree the nearest to the input segment.

        >>> segments = [((0, index), (index, index)) for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> tree.nearest_item(((0, 0), (10, 0))) == (0, ((0, 1), (1, 1)))
        True
        """
        queue = [(0, 0, self._root)]
        while queue:
            _, _, node = heappop(queue)
            for child in node.children:
                heappush(queue,
                         (child.distance_to_segment(segment),
                          -child.index - 1 if child.is_leaf else child.index,
                          child))
            if queue and queue[0][1] < 0:
                _, _, node = heappop(queue)
                return node.item

    def nearest_segment(self, segment: Segment) -> Segment:
        """
        Searches for segment in the tree the nearest to the given segment.

        Time complexity:
            ``O(max_children * log size)``
        Memory complexity:
            ``O(max_children * log size)``

        where ``size = len(self.segments)``,
        ``max_children = self.max_children``.

        :param segment: input segment.
        :returns: segment in the tree the nearest to the input segment.

        >>> segments = [((0, index), (index, index)) for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> tree.nearest_segment(((0, 0), (10, 0))) == ((0, 1), (1, 1))
        True
        """
        _, result = self.nearest_item(segment)
        return result

    def nearest_to_point_index(self, point: Point) -> int:
        """
        Searches for index of segment in the tree
        the nearest to the given point.

        Time complexity:
            ``O(max_children * log size)``
        Memory complexity:
            ``O(max_children * log size)``

        where ``size = len(self.segments)``,
        ``max_children = self.max_children``.

        :param point: input point.
        :returns: index of segment in the tree the nearest to the input point.

        >>> segments = [((0, index), (index, index)) for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> tree.nearest_to_point_index((0, 0)) == 0
        True
        """
        result, _ = self.nearest_to_point_item(point)
        return result

    def nearest_to_point_item(self, point: Point) -> Item:
        """
        Searches for index with segment in the tree
        the nearest to the given point.

        Time complexity:
            ``O(max_children * log size)``
        Memory complexity:
            ``O(max_children * log size)``

        where ``size = len(self.segments)``,
        ``max_children = self.max_children``.

        :param point: input point.
        :returns:
            index with segment in the tree the nearest to the input point.

        >>> segments = [((0, index), (index, index)) for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> tree.nearest_to_point_item((0, 0)) == (0, ((0, 1), (1, 1)))
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

    def nearest_to_point_segment(self, point: Point) -> Segment:
        """
        Searches for segment in the tree the nearest to the given point.

        Time complexity:
            ``O(max_children * log size)``
        Memory complexity:
            ``O(max_children * log size)``

        where ``size = len(self.segments)``,
        ``max_children = self.max_children``.

        :param point: input point.
        :returns: segment in the tree the nearest to the input point.

        >>> segments = [((0, index), (index, index)) for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> tree.nearest_to_point_segment((0, 0)) == ((0, 1), (1, 1))
        True
        """
        _, result = self.nearest_to_point_item(point)
        return result

    def _n_nearest_items(self, n: int, segment: Segment) -> Iterator[Item]:
        queue = [(0, 0, self._root)]
        while n and queue:
            _, _, node = heappop(queue)
            for child in node.children:
                heappush(queue,
                         (child.distance_to_segment(segment),
                          -child.index - 1 if child.is_leaf else child.index,
                          child))
            while n and queue and queue[0][1] < 0:
                _, _, node = heappop(queue)
                yield node.item
                n -= 1

    def _n_nearest_to_point_items(self, n: int, point: Point
                                  ) -> Iterator[Item]:
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


def _create_root(segments: Sequence[Segment],
                 max_children: int,
                 node_cls: Type[Node] = Node) -> Node:
    intervals = [((start_x, end_x)
                  if start_x < end_x
                  else (end_x, start_x),
                  (start_y, end_y)
                  if start_y < end_y
                  else (end_y, start_y))
                 for (start_x, start_y), (end_x, end_y) in segments]
    nodes = [node_cls(index, interval, segment, None)
             for index, (interval, segment) in enumerate(zip(intervals,
                                                             segments))]
    root_interval = reduce(_interval.merge, intervals)
    leaves_count = len(nodes)
    if leaves_count <= max_children:
        # only one node, skip sorting and just fill the root box
        return node_cls(len(nodes), root_interval, None, nodes)
    else:
        (root_min_x, root_max_x), (root_min_y, root_max_y) = root_interval

        def node_key(node: Node,
                     double_root_delta_x: Coordinate
                     = 2 * (root_max_x - root_min_x) or 1,
                     double_root_delta_y: Coordinate
                     = 2 * (root_max_y - root_min_y) or 1,
                     double_root_min_x: Coordinate = 2 * root_min_x,
                     double_root_min_y: Coordinate = 2 * root_min_y) -> int:
            (min_x, max_x), (min_y, max_y) = node.interval
            return _hilbert.index(floor(_hilbert.MAX_COORDINATE
                                        * (min_x + max_x - double_root_min_x)
                                        / double_root_delta_x),
                                  floor(_hilbert.MAX_COORDINATE
                                        * (min_y + max_y - double_root_min_y)
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
                nodes.append(node_cls(len(nodes),
                                      reduce(_interval.merge,
                                             [child.interval
                                              for child in children]),
                                      None,
                                      children))
                start = stop
        return nodes[-1]
