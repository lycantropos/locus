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

from ground.coordinates import (to_divider as _to_divider,
                                to_square_rooter as _to_square_rooter)
from ground.functions import to_dot_producer as _to_dot_producer
from ground.geometries import to_point_cls as _to_point_cls
from ground.hints import (Coordinate,
                          Point,
                          Segment)
from ground.linear import to_segments_relater as _to_segments_relater
from reprit.base import generate_repr

from .core import (hilbert as _hilbert,
                   interval as _interval,
                   segment as _segment)
from .core.utils import ceil_division
from .hints import Interval

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

Item = Tuple[int, Segment]


class Node(Protocol):
    """
    Interface of segmental *R*-tree node.

    Can be implemented for custom metrics definition.
    """

    def __new__(cls,
                index: int,
                interval: Interval,
                segment: Optional[Segment],
                children: Optional[Sequence['Node']]) -> 'Node':
        """Creates node."""

    @property
    def children(self) -> Sequence['Node']:
        """Returns children of the node."""

    @property
    def index(self) -> int:
        """Returns index of the node."""

    @property
    def is_leaf(self) -> bool:
        """Checks whether the node is a leaf."""

    @property
    def item(self) -> Item:
        """Returns underlying index with segment."""

    def distance_to_point(self, point: Point) -> Coordinate:
        """Calculates distance to given point."""

    def distance_to_segment(self, segment: Segment) -> Coordinate:
        """Calculates distance to given segment."""


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
                 node_cls: Optional[Type[Node]] = None) -> None:
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
        self._root = _create_root(segments, max_children,
                                  _to_default_node_cls()
                                  if node_cls is None
                                  else node_cls)

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

        >>> from ground.geometries import to_point_cls, to_segment_cls
        >>> Point, Segment = to_point_cls(), to_segment_cls()
        >>> segments = [Segment(Point(0, index), Point(index, index))
        ...             for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> (tree.n_nearest_indices(2, Segment(Point(0, 0), Point(10, 0)))
        ...  == [0, 1])
        True
        >>> (tree.n_nearest_indices(10, Segment(Point(0, 0), Point(10, 0)))
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

        >>> from ground.geometries import to_point_cls, to_segment_cls
        >>> Point, Segment = to_point_cls(), to_segment_cls()
        >>> segments = [Segment(Point(0, index), Point(index, index))
        ...             for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> (tree.n_nearest_items(2, Segment(Point(0, 0), Point(10, 0)))
        ...  == [(0, Segment(Point(0, 1), Point(1, 1))),
        ...      (1, Segment(Point(0, 2), Point(2, 2)))])
        True
        >>> (tree.n_nearest_items(10, Segment(Point(0, 0), Point(10, 0)))
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

        >>> from ground.geometries import to_point_cls, to_segment_cls
        >>> Point, Segment = to_point_cls(), to_segment_cls()
        >>> segments = [Segment(Point(0, index), Point(index, index))
        ...             for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> (tree.n_nearest_segments(2, Segment(Point(0, 0), Point(10, 0)))
        ...  == [Segment(Point(0, 1), Point(1, 1)),
        ...      Segment(Point(0, 2), Point(2, 2))])
        True
        >>> (tree.n_nearest_segments(10, Segment(Point(0, 0), Point(10, 0)))
        ...  == segments)
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

        >>> from ground.geometries import to_point_cls, to_segment_cls
        >>> Point, Segment = to_point_cls(), to_segment_cls()
        >>> segments = [Segment(Point(0, index), Point(index, index))
        ...             for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> tree.n_nearest_to_point_indices(2, Point(0, 0)) == [0, 1]
        True
        >>> (tree.n_nearest_to_point_indices(10, Point(0, 0))
        ...  == range(len(segments)))
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

        >>> from ground.geometries import to_point_cls, to_segment_cls
        >>> Point, Segment = to_point_cls(), to_segment_cls()
        >>> segments = [Segment(Point(0, index), Point(index, index))
        ...             for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> (tree.n_nearest_to_point_items(2, Point(0, 0))
        ...  == [(0, Segment(Point(0, 1), Point(1, 1))),
        ...      (1, Segment(Point(0, 2), Point(2, 2)))])
        True
        >>> (tree.n_nearest_to_point_items(10, Point(0, 0))
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

        >>> from ground.geometries import to_point_cls, to_segment_cls
        >>> Point, Segment = to_point_cls(), to_segment_cls()
        >>> segments = [Segment(Point(0, index), Point(index, index))
        ...             for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> (tree.n_nearest_to_point_segments(2, Point(0, 0))
        ...  == [Segment(Point(0, 1), Point(1, 1)),
        ...      Segment(Point(0, 2), Point(2, 2))])
        True
        >>> tree.n_nearest_to_point_segments(10, Point(0, 0)) == segments
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

        >>> from ground.geometries import to_point_cls, to_segment_cls
        >>> Point, Segment = to_point_cls(), to_segment_cls()
        >>> segments = [Segment(Point(0, index), Point(index, index))
        ...             for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> tree.nearest_index(Segment(Point(0, 0), Point(10, 0))) == 0
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

        >>> from ground.geometries import to_point_cls, to_segment_cls
        >>> Point, Segment = to_point_cls(), to_segment_cls()
        >>> segments = [Segment(Point(0, index), Point(index, index))
        ...             for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> (tree.nearest_item(Segment(Point(0, 0), Point(10, 0)))
        ...  == (0, Segment(Point(0, 1), Point(1, 1))))
        True
        """
        queue = [(0, 0, self._root)]
        while queue:
            _, _, node = heappop(queue)
            for child in node.children:
                heappush(queue,
                         (child.distance_to_segment(segment),
                          child.index if child.is_leaf else -child.index - 1,
                          child))
            if queue and queue[0][1] >= 0:
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

        >>> from ground.geometries import to_point_cls, to_segment_cls
        >>> Point, Segment = to_point_cls(), to_segment_cls()
        >>> segments = [Segment(Point(0, index), Point(index, index))
        ...             for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> (tree.nearest_segment(Segment(Point(0, 0), Point(10, 0)))
        ...  == Segment(Point(0, 1), Point(1, 1)))
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

        >>> from ground.geometries import to_point_cls, to_segment_cls
        >>> Point, Segment = to_point_cls(), to_segment_cls()
        >>> segments = [Segment(Point(0, index), Point(index, index))
        ...             for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> tree.nearest_to_point_index(Point(0, 0)) == 0
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

        >>> from ground.geometries import to_point_cls, to_segment_cls
        >>> Point, Segment = to_point_cls(), to_segment_cls()
        >>> segments = [Segment(Point(0, index), Point(index, index))
        ...             for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> (tree.nearest_to_point_item(Point(0, 0))
        ...  == (0, Segment(Point(0, 1), Point(1, 1))))
        True
        """
        queue = [(0, 0, self._root)]
        while queue:
            _, _, node = heappop(queue)
            for child in node.children:
                heappush(queue,
                         (child.distance_to_point(point),
                          child.index if child.is_leaf else -child.index - 1,
                          child))
            if queue and queue[0][1] >= 0:
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

        >>> from ground.geometries import to_point_cls, to_segment_cls
        >>> Point, Segment = to_point_cls(), to_segment_cls()
        >>> segments = [Segment(Point(0, index), Point(index, index))
        ...             for index in range(1, 11)]
        >>> tree = Tree(segments)
        >>> (tree.nearest_to_point_segment(Point(0, 0))
        ...  == Segment(Point(0, 1), Point(1, 1)))
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
                          child.index if child.is_leaf else -child.index - 1,
                          child))
            while n and queue and queue[0][1] >= 0:
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
                          child.index if child.is_leaf else -child.index - 1,
                          child))
            while n and queue and queue[0][1] >= 0:
                _, _, node = heappop(queue)
                yield node.item
                n -= 1


def _create_root(segments: Sequence[Segment],
                 max_children: int,
                 node_cls: Type[Node]) -> Node:
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
                                      None, children))
                start = stop
        return nodes[-1]


def _to_default_node_cls() -> Type[Node]:
    class Node:
        __slots__ = 'index', 'interval', 'segment', 'children'

        def __init__(self,
                     index: int,
                     interval: Interval,
                     segment: Optional[Segment],
                     children: Optional[Sequence['Node']]) -> None:
            self.index = index
            self.interval = interval
            self.segment = segment
            self.children = children

        __repr__ = generate_repr(__init__)

        _divider = staticmethod(_to_divider())
        _dot_producer = staticmethod(_to_dot_producer())
        _point_cls = staticmethod(_to_point_cls())
        _segments_relater = staticmethod(_to_segments_relater())
        _square_rooter = staticmethod(_to_square_rooter())

        @property
        def is_leaf(self) -> bool:
            return self.children is None

        @property
        def item(self) -> Item:
            return self.index, self.segment

        def distance_to_point(self, point: Point,
                              *,
                              _minus_inf: Coordinate = -inf) -> Coordinate:
            return (_segment.distance_to_point(
                    self._divider, self._dot_producer, self._square_rooter,
                    self.segment, point) or _minus_inf
                    if self.is_leaf
                    else _interval.distance_to_point(self.interval, point))

        def distance_to_segment(self, segment: Segment,
                                *,
                                _minus_inf: Coordinate = -inf) -> Coordinate:
            return (_segment.distance_to(
                    self._divider, self._dot_producer, self._segments_relater,
                    self._square_rooter, self.segment.start,
                    self.segment.end, segment.start, segment.end) or _minus_inf
                    if self.is_leaf
                    else _segment.distance_to_interval(
                    self._divider, self._dot_producer, self._point_cls,
                    self._segments_relater, self._square_rooter, segment,
                    self.interval))

    return Node
