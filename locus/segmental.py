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

from ground.base import (Context as _Context,
                         get_context as _get_context)
from ground.hints import (Box,
                          Coordinate,
                          Point,
                          Segment)
from reprit.base import generate_repr

from .core import (box as _box,
                   hilbert as _hilbert,
                   segment as _segment)
from .core.utils import ceil_division

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
                box: Box,
                segment: Optional[Segment],
                children: Optional[Sequence['Node']]) -> 'Node':
        """Creates node."""

    @property
    def box(self) -> Box:
        """Returns box of the node."""

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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point, Segment = context.point_cls, context.segment_cls
        >>> segments = [Segment(Point(0, index), Point(index, index))
        ...             for index in range(1, 11)]
        >>> tree = Tree(segments)
        """
        self._segments = segments
        self._max_children = max_children
        context = _get_context()
        self._root = _create_root(segments, max_children,
                                  _to_default_node_cls(context)
                                  if node_cls is None
                                  else node_cls,
                                  context)

    __repr__ = generate_repr(__init__)

    @property
    def segments(self) -> Sequence[Segment]:
        """
        Returns underlying segments.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point, Segment = context.point_cls, context.segment_cls
        >>> segments = [Segment(Point(0, index), Point(index, index))
        ...             for index in range(1, 11)]
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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point, Segment = context.point_cls, context.segment_cls
        >>> segments = [Segment(Point(0, index), Point(index, index))
        ...             for index in range(1, 11)]
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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point, Segment = context.point_cls, context.segment_cls
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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point, Segment = context.point_cls, context.segment_cls
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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point, Segment = context.point_cls, context.segment_cls
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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point, Segment = context.point_cls, context.segment_cls
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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point, Segment = context.point_cls, context.segment_cls
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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point, Segment = context.point_cls, context.segment_cls
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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point, Segment = context.point_cls, context.segment_cls
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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point, Segment = context.point_cls, context.segment_cls
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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point, Segment = context.point_cls, context.segment_cls
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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point, Segment = context.point_cls, context.segment_cls
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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point, Segment = context.point_cls, context.segment_cls
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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point, Segment = context.point_cls, context.segment_cls
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
                 node_cls: Type[Node],
                 context: _Context) -> Node:
    box_cls, merge_boxes = context.box_cls, context.merged_box
    boxes = [box_cls(*((segment.start.x, segment.end.x)
                       if segment.start.x < segment.end.x
                       else (segment.end.x, segment.start.x)),
                     *((segment.start.y, segment.end.y)
                       if segment.start.y < segment.end.y
                       else (segment.end.y, segment.start.y)))
             for segment in segments]
    nodes = [node_cls(index, box, segment, None)
             for index, (box, segment) in enumerate(zip(boxes, segments))]
    root_box = reduce(merge_boxes, boxes)
    leaves_count = len(nodes)
    if leaves_count <= max_children:
        # only one node, skip sorting and just fill the root box
        return node_cls(len(nodes), root_box, None, nodes)
    else:
        def node_key(node: Node,
                     double_root_delta_x: Coordinate
                     = 2 * (root_box.max_x - root_box.min_x) or 1,
                     double_root_delta_y: Coordinate
                     = 2 * (root_box.max_y - root_box.min_y) or 1,
                     double_root_min_x: Coordinate = 2 * root_box.min_x,
                     double_root_min_y: Coordinate = 2 * root_box.min_y
                     ) -> int:
            box = node.box
            return _hilbert.index(floor(_hilbert.MAX_COORDINATE
                                        * (box.min_x + box.max_x
                                           - double_root_min_x)
                                        / double_root_delta_x),
                                  floor(_hilbert.MAX_COORDINATE
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
                nodes.append(node_cls(len(nodes),
                                      reduce(merge_boxes,
                                             [child.box
                                              for child in children]),
                                      None, children))
                start = stop
        return nodes[-1]


def _to_default_node_cls(context: _Context) -> Type[Node]:
    class Node:
        __slots__ = 'index', 'box', 'segment', 'children'

        def __init__(self,
                     index: int,
                     box: Box,
                     segment: Optional[Segment],
                     children: Optional[Sequence['Node']]) -> None:
            self.index, self.box, self.segment, self.children = (
                index, box, segment, children)

        __repr__ = generate_repr(__init__)

        _dot_product = staticmethod(context.dot_product)
        _point_cls = staticmethod(context.point_cls)
        _segments_relation = staticmethod(context.segments_relation)

        @property
        def is_leaf(self) -> bool:
            return self.children is None

        @property
        def item(self) -> Item:
            return self.index, self.segment

        def distance_to_point(self, point: Point,
                              *,
                              _minus_inf: Coordinate = -inf) -> Coordinate:
            return (_segment.distance_to_point(self._dot_product,
                                               self.segment.start,
                                               self.segment.end, point)
                    or _minus_inf
                    if self.is_leaf
                    else _box.distance_to_point(self.box, point))

        def distance_to_segment(self, segment: Segment,
                                *,
                                _minus_inf: Coordinate = -inf) -> Coordinate:
            return (_segment.distance_to(self._dot_product,
                                         self._segments_relation,
                                         self.segment.start, self.segment.end,
                                         segment.start, segment.end)
                    or _minus_inf
                    if self.is_leaf
                    else _segment.distance_to_box(self._dot_product,
                                                  self._point_cls,
                                                  self._segments_relation,
                                                  segment, self.box))

    return Node
