from collections.abc import Iterator as _Iterator, Sequence as _Sequence
from heapq import heappop as _heappop, heappush as _heappush
from typing import Generic as _Generic

from ground.context import Context as _Context
from ground.hints import Point as _Point, Segment as _Segment
from reprit.base import generate_repr as _generate_repr

from ._core.hints import HasCustomRepr as _HasCustomRepr, ScalarT as _ScalarT
from ._core.segmental import (
    AnyNode as _AnyNode,
    Item as _Item,
    create_root as _create_root,
    is_leaf as _is_leaf,
)


class Tree(_HasCustomRepr, _Generic[_ScalarT]):
    """
    Represents packed 2-dimensional segmental Hilbert *R*-tree.

    Reference:
        https://en.wikipedia.org/wiki/Hilbert_R-tree#Packed_Hilbert_R-trees
    """

    __slots__ = '_context', '_max_children', '_root', '_segments'

    def __init__(
        self,
        segments: _Sequence[_Segment[_ScalarT]],
        /,
        *,
        context: _Context[_ScalarT],
        max_children: int = 16,
    ) -> None:
        """
        Initializes tree from segments.

        Time complexity:
            ``O(size * log size)``
        Memory complexity:
            ``O(size)``

        where ``size = len(segments)``.
        """
        box_cls = context.box_cls
        self._context, self._max_children, self._root, self._segments = (
            context,
            max_children,
            _create_root(
                segments,
                [
                    box_cls(
                        *(
                            (segment.start.x, segment.end.x)
                            if segment.start.x < segment.end.x
                            else (segment.end.x, segment.start.x)
                        ),
                        *(
                            (segment.start.y, segment.end.y)
                            if segment.start.y < segment.end.y
                            else (segment.end.y, segment.start.y)
                        ),
                    )
                    for segment in segments
                ],
                max_children,
                box_point_metric=context.box_point_squared_distance,
                box_segment_metric=context.box_segment_squared_distance,
                boxes_merger=context.merged_box,
                coordinate_factory=context.coordinate_factory,
                segment_point_metric=context.segment_point_squared_distance,
                segments_metric=context.segments_squared_distance,
                zero=context.zero,
            ),
            segments,
        )

    __repr__ = _generate_repr(__init__)

    @property
    def context(self, /) -> _Context[_ScalarT]:
        """
        Returns context of the tree.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``
        """
        return self._context

    @property
    def max_children(self, /) -> int:
        """
        Returns maximum number of children in each node.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Point, Segment = context.point_cls, context.segment_cls
        >>> segments = [
        ...     Segment(Point(0, index), Point(index, index))
        ...     for index in range(1, 11)
        ... ]
        >>> tree = Tree(segments, context=context)
        >>> tree.max_children == 16
        True
        """
        return self._max_children

    @property
    def segments(self, /) -> _Sequence[_Segment[_ScalarT]]:
        """
        Returns underlying segments.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Point, Segment = context.point_cls, context.segment_cls
        >>> segments = [
        ...     Segment(Point(0, index), Point(index, index))
        ...     for index in range(1, 11)
        ... ]
        >>> tree = Tree(segments, context=context)
        >>> tree.segments == segments
        True
        """
        return self._segments

    def n_nearest_indices(
        self, n: int, segment: _Segment[_ScalarT], /
    ) -> _Sequence[int]:
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

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Point, Segment = context.point_cls, context.segment_cls
        >>> segments = [
        ...     Segment(Point(0, index), Point(index, index))
        ...     for index in range(1, 11)
        ... ]
        >>> tree = Tree(segments, context=context)
        >>> (
        ...     tree.n_nearest_indices(2, Segment(Point(0, 0), Point(10, 0)))
        ...     == [0, 1]
        ... )
        True
        >>> (
        ...     tree.n_nearest_indices(10, Segment(Point(0, 0), Point(10, 0)))
        ...     == range(len(segments))
        ... )
        True
        """
        return (
            [index for index, _ in self._n_nearest_items(n, segment)]
            if n < len(self._segments)
            else range(len(self._segments))
        )

    def n_nearest_items(
        self, n: int, segment: _Segment[_ScalarT], /
    ) -> _Sequence[_Item[_ScalarT]]:
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

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Point, Segment = context.point_cls, context.segment_cls
        >>> segments = [
        ...     Segment(Point(0, index), Point(index, index))
        ...     for index in range(1, 11)
        ... ]
        >>> tree = Tree(segments, context=context)
        >>> (
        ...     tree.n_nearest_items(2, Segment(Point(0, 0), Point(10, 0)))
        ...     == [
        ...         (0, Segment(Point(0, 1), Point(1, 1))),
        ...         (1, Segment(Point(0, 2), Point(2, 2))),
        ...     ]
        ... )
        True
        >>> (
        ...     tree.n_nearest_items(10, Segment(Point(0, 0), Point(10, 0)))
        ...     == list(enumerate(segments))
        ... )
        True
        """
        return list(
            self._n_nearest_items(n, segment)
            if n < len(self._segments)
            else enumerate(self._segments)
        )

    def n_nearest_segments(
        self, n: int, segment: _Segment[_ScalarT], /
    ) -> _Sequence[_Segment[_ScalarT]]:
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

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Point, Segment = context.point_cls, context.segment_cls
        >>> segments = [
        ...     Segment(Point(0, index), Point(index, index))
        ...     for index in range(1, 11)
        ... ]
        >>> tree = Tree(segments, context=context)
        >>> (
        ...     tree.n_nearest_segments(2, Segment(Point(0, 0), Point(10, 0)))
        ...     == [
        ...         Segment(Point(0, 1), Point(1, 1)),
        ...         Segment(Point(0, 2), Point(2, 2)),
        ...     ]
        ... )
        True
        >>> (
        ...     tree.n_nearest_segments(10, Segment(Point(0, 0), Point(10, 0)))
        ...     == segments
        ... )
        True
        """
        return (
            [segment for _, segment in self._n_nearest_items(n, segment)]
            if n < len(self._segments)
            else self._segments
        )

    def n_nearest_to_point_indices(
        self, n: int, point: _Point[_ScalarT], /
    ) -> _Sequence[int]:
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

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Point, Segment = context.point_cls, context.segment_cls
        >>> segments = [
        ...     Segment(Point(0, index), Point(index, index))
        ...     for index in range(1, 11)
        ... ]
        >>> tree = Tree(segments, context=context)
        >>> tree.n_nearest_to_point_indices(2, Point(0, 0)) == [0, 1]
        True
        >>> (
        ...     tree.n_nearest_to_point_indices(10, Point(0, 0))
        ...     == range(len(segments))
        ... )
        True
        """
        return (
            [index for index, _ in self._n_nearest_to_point_items(n, point)]
            if n < len(self._segments)
            else range(len(self._segments))
        )

    def n_nearest_to_point_items(
        self, n: int, point: _Point[_ScalarT], /
    ) -> _Sequence[_Item[_ScalarT]]:
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

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Point, Segment = context.point_cls, context.segment_cls
        >>> segments = [
        ...     Segment(Point(0, index), Point(index, index))
        ...     for index in range(1, 11)
        ... ]
        >>> tree = Tree(segments, context=context)
        >>> (
        ...     tree.n_nearest_to_point_items(2, Point(0, 0))
        ...     == [
        ...         (0, Segment(Point(0, 1), Point(1, 1))),
        ...         (1, Segment(Point(0, 2), Point(2, 2))),
        ...     ]
        ... )
        True
        >>> (
        ...     tree.n_nearest_to_point_items(10, Point(0, 0))
        ...     == list(enumerate(segments))
        ... )
        True
        """
        return list(
            self._n_nearest_to_point_items(n, point)
            if n < len(self._segments)
            else enumerate(self._segments)
        )

    def n_nearest_to_point_segments(
        self, n: int, point: _Point[_ScalarT], /
    ) -> _Sequence[_Segment[_ScalarT]]:
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

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Point, Segment = context.point_cls, context.segment_cls
        >>> segments = [
        ...     Segment(Point(0, index), Point(index, index))
        ...     for index in range(1, 11)
        ... ]
        >>> tree = Tree(segments, context=context)
        >>> (
        ...     tree.n_nearest_to_point_segments(2, Point(0, 0))
        ...     == [
        ...         Segment(Point(0, 1), Point(1, 1)),
        ...         Segment(Point(0, 2), Point(2, 2)),
        ...     ]
        ... )
        True
        >>> tree.n_nearest_to_point_segments(10, Point(0, 0)) == segments
        True
        """
        return (
            [
                segment
                for _, segment in self._n_nearest_to_point_items(n, point)
            ]
            if n < len(self._segments)
            else self._segments
        )

    def nearest_index(self, segment: _Segment[_ScalarT], /) -> int:
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

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Point, Segment = context.point_cls, context.segment_cls
        >>> segments = [
        ...     Segment(Point(0, index), Point(index, index))
        ...     for index in range(1, 11)
        ... ]
        >>> tree = Tree(segments, context=context)
        >>> tree.nearest_index(Segment(Point(0, 0), Point(10, 0))) == 0
        True
        """
        result, _ = self.nearest_item(segment)
        return result

    def nearest_item(self, segment: _Segment[_ScalarT], /) -> _Item[_ScalarT]:
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

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Point, Segment = context.point_cls, context.segment_cls
        >>> segments = [
        ...     Segment(Point(0, index), Point(index, index))
        ...     for index in range(1, 11)
        ... ]
        >>> tree = Tree(segments, context=context)
        >>> (
        ...     tree.nearest_item(Segment(Point(0, 0), Point(10, 0)))
        ...     == (0, Segment(Point(0, 1), Point(1, 1)))
        ... )
        True
        """
        queue: list[tuple[_ScalarT, int, _AnyNode[_ScalarT]]] = [
            (self._context.zero, 0, self._root)
        ]
        while queue:
            _, _, node = _heappop(queue)
            assert not _is_leaf(node), node
            for child in node.children:
                _heappush(
                    queue,
                    (
                        child.distance_to_segment(segment),
                        child.index if _is_leaf(child) else -child.index - 1,
                        child,
                    ),
                )
            if queue and queue[0][1] >= 0:
                _, _, node = _heappop(queue)
                assert _is_leaf(node), node
                return node.item
        raise ValueError

    def nearest_segment(
        self, segment: _Segment[_ScalarT], /
    ) -> _Segment[_ScalarT]:
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

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Point, Segment = context.point_cls, context.segment_cls
        >>> segments = [
        ...     Segment(Point(0, index), Point(index, index))
        ...     for index in range(1, 11)
        ... ]
        >>> tree = Tree(segments, context=context)
        >>> (
        ...     tree.nearest_segment(Segment(Point(0, 0), Point(10, 0)))
        ...     == Segment(Point(0, 1), Point(1, 1))
        ... )
        True
        """
        _, result = self.nearest_item(segment)
        return result

    def nearest_to_point_index(self, point: _Point[_ScalarT], /) -> int:
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

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Point, Segment = context.point_cls, context.segment_cls
        >>> segments = [
        ...     Segment(Point(0, index), Point(index, index))
        ...     for index in range(1, 11)
        ... ]
        >>> tree = Tree(segments, context=context)
        >>> tree.nearest_to_point_index(Point(0, 0)) == 0
        True
        """
        result, _ = self.nearest_to_point_item(point)
        return result

    def nearest_to_point_item(
        self, point: _Point[_ScalarT], /
    ) -> _Item[_ScalarT]:
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

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Point, Segment = context.point_cls, context.segment_cls
        >>> segments = [
        ...     Segment(Point(0, index), Point(index, index))
        ...     for index in range(1, 11)
        ... ]
        >>> tree = Tree(segments, context=context)
        >>> (
        ...     tree.nearest_to_point_item(Point(0, 0))
        ...     == (0, Segment(Point(0, 1), Point(1, 1)))
        ... )
        True
        """
        queue: list[tuple[_ScalarT, int, _AnyNode[_ScalarT]]] = [
            (self._context.zero, 0, self._root)
        ]
        while queue:
            _, _, node = _heappop(queue)
            assert not _is_leaf(node), node
            for child in node.children:
                _heappush(
                    queue,
                    (
                        child.distance_to_point(point),
                        child.index if _is_leaf(child) else -child.index - 1,
                        child,
                    ),
                )
            if queue and queue[0][1] >= 0:
                _, _, node = _heappop(queue)
                assert _is_leaf(node), node
                return node.item
        raise ValueError

    def nearest_to_point_segment(
        self, point: _Point[_ScalarT], /
    ) -> _Segment[_ScalarT]:
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

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Point, Segment = context.point_cls, context.segment_cls
        >>> segments = [
        ...     Segment(Point(0, index), Point(index, index))
        ...     for index in range(1, 11)
        ... ]
        >>> tree = Tree(segments, context=context)
        >>> (
        ...     tree.nearest_to_point_segment(Point(0, 0))
        ...     == Segment(Point(0, 1), Point(1, 1))
        ... )
        True
        """
        _, result = self.nearest_to_point_item(point)
        return result

    def _n_nearest_items(
        self, n: int, segment: _Segment[_ScalarT], /
    ) -> _Iterator[_Item[_ScalarT]]:
        queue: list[tuple[_ScalarT, int, _AnyNode[_ScalarT]]] = [
            (self._context.zero, 0, self._root)
        ]
        while n and queue:
            _, _, node = _heappop(queue)
            assert not _is_leaf(node), node
            for child in node.children:
                _heappush(
                    queue,
                    (
                        child.distance_to_segment(segment),
                        child.index if _is_leaf(child) else -child.index - 1,
                        child,
                    ),
                )
            while n and queue and queue[0][1] >= 0:
                _, _, node = _heappop(queue)
                assert _is_leaf(node), node
                yield node.item
                n -= 1

    def _n_nearest_to_point_items(
        self, n: int, point: _Point[_ScalarT], /
    ) -> _Iterator[_Item[_ScalarT]]:
        queue: list[tuple[_ScalarT, int, _AnyNode[_ScalarT]]] = [
            (self._context.zero, 0, self._root)
        ]
        while n and queue:
            _, _, node = _heappop(queue)
            assert not _is_leaf(node), node
            for child in node.children:
                _heappush(
                    queue,
                    (
                        child.distance_to_point(point),
                        child.index if _is_leaf(child) else -child.index - 1,
                        child,
                    ),
                )
            while n and queue and queue[0][1] >= 0:
                _, _, node = _heappop(queue)
                assert _is_leaf(node), node
                yield node.item
                n -= 1
