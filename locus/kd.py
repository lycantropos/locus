from collections.abc import Iterator as _Iterator, Sequence as _Sequence
from heapq import heappush as _heappush, heapreplace as _heapreplace
from typing import Generic as _Generic

from ground.context import Context as _Context
from ground.hints import Box as _Box, Point as _Point
from reprit.base import generate_repr as _generate_repr

from ._core import box as _box
from ._core.hints import HasCustomRepr as _HasCustomRepr, ScalarT as _ScalarT
from ._core.kd import (
    Item as _Item,
    NIL as _NIL,
    Node as _Node,
    create_node as _create_node,
)


class Tree(_HasCustomRepr, _Generic[_ScalarT]):
    """
    Represents `k`-dimensional (aka *kd*) tree.

    Reference:
        https://en.wikipedia.org/wiki/K-d_tree
    """

    __slots__ = '_context', '_points', '_root'

    def __init__(
        self,
        points: _Sequence[_Point[_ScalarT]],
        /,
        *,
        context: _Context[_ScalarT],
    ) -> None:
        """
        Initializes tree from points.

        Time complexity:
            ``O(dimension * size * log size)``
        Memory complexity:
            ``O(dimension * size)``

        where ``dimension = len(points[0])``, ``size = len(points)``.
        """
        self._context, self._points, self._root = (
            context,
            points,
            _create_node(
                range(len(points)),
                points,
                is_y_axis=False,
                metric=context.points_squared_distance,
            ),
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
    def points(self, /) -> _Sequence[_Point[_ScalarT]]:
        """
        Returns underlying points.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Point = context.point_cls
        >>> points = list(map(Point, range(-5, 6), range(10)))
        >>> tree = Tree(points, context=context)
        >>> tree.points == points
        True
        """
        return self._points

    def n_nearest_indices(
        self, n: int, point: _Point[_ScalarT], /
    ) -> _Sequence[int]:
        """
        Searches for indices of points in the tree
        that are the nearest to the given point.

        Time complexity:
            ``O(min(n, size) * log size)``
        Memory complexity:
            ``O(min(n, size) * log size)``

        where ``size = len(self.points)``.

        Reference:
            https://en.wikipedia.org/wiki/K-d_tree#Nearest_neighbour_search

        :param n: positive upper bound for number of result indices.
        :param point: input point.
        :returns: indices of points in the tree the nearest to the input point.

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Point = context.point_cls
        >>> points = list(map(Point, range(-5, 6), range(10)))
        >>> tree = Tree(points, context=context)
        >>> tree.n_nearest_indices(2, Point(0, 0)) == [2, 3]
        True
        >>> (
        ...     tree.n_nearest_indices(len(points), Point(0, 0))
        ...     == range(len(points))
        ... )
        True
        """
        return (
            [index for index, _ in self._n_nearest_items(n, point)]
            if n < len(self._points)
            else range(len(self._points))
        )

    def n_nearest_points(
        self, n: int, point: _Point[_ScalarT], /
    ) -> _Sequence[_Point[_ScalarT]]:
        """
        Searches for points in the tree the nearest to the given point.

        Time complexity:
            ``O(min(n, size) * log size)``
        Memory complexity:
            ``O(min(n, size) * log size)``

        where ``size = len(self.points)``.

        Reference:
            https://en.wikipedia.org/wiki/K-d_tree#Nearest_neighbour_search

        :param n: positive upper bound for number of result points.
        :param point: input point.
        :returns: points in the tree the nearest to the input point.

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Point = context.point_cls
        >>> points = list(map(Point, range(-5, 6), range(10)))
        >>> tree = Tree(points, context=context)
        >>> (
        ...     tree.n_nearest_points(2, Point(0, 0))
        ...     == [Point(-3, 2), Point(-2, 3)]
        ... )
        True
        >>> tree.n_nearest_points(len(points), Point(0, 0)) == points
        True
        """
        return (
            [point for _, point in self._n_nearest_items(n, point)]
            if n < len(self._points)
            else self._points
        )

    def n_nearest_items(
        self, n: int, point: _Point[_ScalarT], /
    ) -> _Sequence[_Item[_ScalarT]]:
        """
        Searches for indices with points in the tree
        that are the nearest to the given point.

        Time complexity:
            ``O(min(n, size) * log size)``
        Memory complexity:
            ``O(min(n, size) * log size)``

        where ``size = len(self.points)``.

        Reference:
            https://en.wikipedia.org/wiki/K-d_tree#Nearest_neighbour_search

        :param n: positive upper bound for number of result indices.
        :param point: input point.
        :returns:
            indices with points in the tree the nearest to the input point.

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Point = context.point_cls
        >>> points = list(map(Point, range(-5, 6), range(10)))
        >>> tree = Tree(points, context=context)
        >>> (
        ...     tree.n_nearest_items(2, Point(0, 0))
        ...     == [(2, Point(-3, 2)), (3, Point(-2, 3))]
        ... )
        True
        >>> (
        ...     tree.n_nearest_items(len(points), Point(0, 0))
        ...     == list(enumerate(points))
        ... )
        True
        """
        return (
            self._n_nearest_items(n, point)
            if n < len(self._points)
            else list(enumerate(self._points))
        )

    def _n_nearest_items(
        self, n: int, point: _Point[_ScalarT], /
    ) -> list[_Item[_ScalarT]]:
        if self._root is _NIL:
            return []
        candidates: list[tuple[_ScalarT, _Item[_ScalarT]]] = []
        queue: list[_Node[_ScalarT]] = [self._root]
        push, pop = queue.append, queue.pop
        while queue:
            node = pop()
            distance_to_point = node.distance_to_point(point)
            candidate = (-distance_to_point, node.item)
            if len(candidates) < n:
                _heappush(candidates, candidate)
            elif distance_to_point < -candidates[0][0]:
                _heapreplace(candidates, candidate)
            coordinate = node.projector(point)
            point_is_on_the_left = coordinate < node.projection
            if point_is_on_the_left:
                if node.left is not _NIL:
                    push(node.left)
            elif node.right is not _NIL:
                push(node.right)
            if len(candidates) < n or (
                node.distance_to_coordinate(coordinate) < -candidates[0][0]
            ):
                if point_is_on_the_left:
                    if node.right is not _NIL:
                        push(node.right)
                elif node.left is not _NIL:
                    push(node.left)
        return [item for _, item in candidates]

    def nearest_index(self, point: _Point[_ScalarT], /) -> int:
        """
        Searches for index of a point in the tree
        that is the nearest to the given point.

        Time complexity:
            ``O(log size)``
        Memory complexity:
            ``O(log size)``

        where ``size = len(self.points)``.

        Reference:
            https://en.wikipedia.org/wiki/K-d_tree#Nearest_neighbour_search

        :param point: input point.
        :returns: index of a point in the tree the nearest to the input point.

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Point = context.point_cls
        >>> points = list(map(Point, range(-5, 6), range(10)))
        >>> tree = Tree(points, context=context)
        >>> tree.nearest_index(Point(0, 0)) == 2
        True
        >>> tree.nearest_index(Point(-3, 2)) == 2
        True
        """
        result, _ = self.nearest_item(point)
        return result

    def nearest_point(self, point: _Point[_ScalarT], /) -> _Point[_ScalarT]:
        """
        Searches for point in the tree that is the nearest to the given point.

        Time complexity:
            ``O(log size)``
        Memory complexity:
            ``O(log size)``

        where ``size = len(self.points)``.

        Reference:
            https://en.wikipedia.org/wiki/K-d_tree#Nearest_neighbour_search

        :param point: input point.
        :returns: point in the tree the nearest to the input point.

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Point = context.point_cls
        >>> points = list(map(Point, range(-5, 6), range(10)))
        >>> tree = Tree(points, context=context)
        >>> tree.nearest_point(Point(0, 0)) == Point(-3, 2)
        True
        >>> tree.nearest_point(Point(-3, 2)) == Point(-3, 2)
        True
        """
        _, result = self.nearest_item(point)
        return result

    def nearest_item(self, point: _Point[_ScalarT], /) -> _Item[_ScalarT]:
        """
        Searches for index with point in the tree
        that is the nearest to the given point.

        Time complexity:
            ``O(log size)``
        Memory complexity:
            ``O(log size)``

        where ``size = len(self.points)``.

        Reference:
            https://en.wikipedia.org/wiki/K-d_tree#Nearest_neighbour_search

        :param point: input point.
        :returns: index with point in the tree the nearest to the input point.

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Point = context.point_cls
        >>> points = list(map(Point, range(-5, 6), range(10)))
        >>> tree = Tree(points, context=context)
        >>> tree.nearest_item(Point(0, 0)) == (2, Point(-3, 2))
        True
        >>> tree.nearest_item(Point(-3, 2)) == (2, Point(-3, 2))
        True
        """
        if self._root is _NIL:
            raise ValueError
        node: _Node[_ScalarT] = self._root
        result, min_distance = node.item, node.distance_to_point(point)
        queue = [node]
        push, pop = queue.append, queue.pop
        while queue:
            node = pop()
            distance_to_point = node.distance_to_point(point)
            if distance_to_point < min_distance:
                result, min_distance = node.item, distance_to_point
            coordinate = node.projector(point)
            point_is_on_the_left = coordinate < node.projection
            if point_is_on_the_left:
                if node.left is not _NIL:
                    push(node.left)
            elif node.right is not _NIL:
                push(node.right)
            if node.distance_to_coordinate(coordinate) < min_distance:
                if point_is_on_the_left:
                    if node.right is not _NIL:
                        push(node.right)
                elif node.left is not _NIL:
                    push(node.left)
        return result

    def find_box_indices(self, box: _Box[_ScalarT], /) -> list[int]:
        """
        Searches for indices of points that lie inside the given box.

        Time complexity:
            ``O(dimension * size ** (1 - 1 / dimension) + hits_count)``
        Memory complexity:
            ``O(dimension * size ** (1 - 1 / dimension) + hits_count)``

        where ``dimension = len(self.points[0])``, ``size = len(self.points)``,
        ``hits_count`` --- number of found indices.

        Reference:
            https://en.wikipedia.org/wiki/K-d_tree#Range_search

        :param box: box to search in.
        :returns: indices of points that lie inside the box.

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Box, Point = context.box_cls, context.point_cls
        >>> points = list(map(Point, range(-5, 6), range(10)))
        >>> tree = Tree(points, context=context)
        >>> tree.find_box_indices(Box(-3, 3, 0, 1)) == []
        True
        >>> tree.find_box_indices(Box(-3, 3, 0, 2)) == [2]
        True
        >>> tree.find_box_indices(Box(-3, 3, 0, 3)) == [2, 3]
        True
        """
        return [index for index, _ in self._find_box_items(box)]

    def find_box_points(
        self, box: _Box[_ScalarT], /
    ) -> list[_Point[_ScalarT]]:
        """
        Searches for points that lie inside the given box.

        Time complexity:
            ``O(dimension * size ** (1 - 1 / dimension) + hits_count)``
        Memory complexity:
            ``O(dimension * size ** (1 - 1 / dimension) + hits_count)``

        where ``dimension = len(self.points[0])``, ``size = len(self.points)``,
        ``hits_count`` --- number of found points.

        Reference:
            https://en.wikipedia.org/wiki/K-d_tree#Range_search

        :param box: box to search in.
        :returns: points that lie inside the box.

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Box, Point = context.box_cls, context.point_cls
        >>> points = list(map(Point, range(-5, 6), range(10)))
        >>> tree = Tree(points, context=context)
        >>> tree.find_box_points(Box(-3, 3, 0, 1)) == []
        True
        >>> tree.find_box_points(Box(-3, 3, 0, 2)) == [Point(-3, 2)]
        True
        >>> (
        ...     tree.find_box_points(Box(-3, 3, 0, 3))
        ...     == [Point(-3, 2), Point(-2, 3)]
        ... )
        True
        """
        return [point for _, point in self._find_box_items(box)]

    def find_box_items(self, box: _Box[_ScalarT], /) -> list[_Item[_ScalarT]]:
        """
        Searches for indices with points in the tree
        that lie inside the given box.

        Time complexity:
            ``O(dimension * size ** (1 - 1 / dimension) + hits_count)``
        Memory complexity:
            ``O(dimension * size ** (1 - 1 / dimension) + hits_count)``

        where ``dimension = len(self.points[0])``, ``size = len(self.points)``,
        ``hits_count`` --- number of found indices with points.

        Reference:
            https://en.wikipedia.org/wiki/K-d_tree#Range_search

        :param box: box to search in.
        :returns: indices with points in the tree that lie inside the box.

        >>> import math
        >>> from fractions import Fraction
        >>> from ground.context import Context
        >>> context = Context(coordinate_factory=Fraction, sqrt=math.sqrt)
        >>> Box, Point = context.box_cls, context.point_cls
        >>> points = list(map(Point, range(-5, 6), range(10)))
        >>> tree = Tree(points, context=context)
        >>> tree.find_box_items(Box(-3, 3, 0, 1)) == []
        True
        >>> tree.find_box_items(Box(-3, 3, 0, 2)) == [(2, Point(-3, 2))]
        True
        >>> (
        ...     tree.find_box_items(Box(-3, 3, 0, 3))
        ...     == [(2, Point(-3, 2)), (3, Point(-2, 3))]
        ... )
        True
        """
        return list(self._find_box_items(box))

    def _find_box_items(
        self, box: _Box[_ScalarT], /
    ) -> _Iterator[_Item[_ScalarT]]:
        if self._root is _NIL:
            return
        queue: list[_Node[_ScalarT]] = [self._root]
        push, pop = queue.append, queue.pop
        while queue:
            node = pop()
            if _box.contains_point(box, node.point):
                yield node.item
            min_coordinate, max_coordinate = (
                (box.min_y, box.max_y)
                if node.is_y_axis
                else (box.min_x, box.max_x)
            )
            coordinate = node.projector(node.point)
            if node.left is not _NIL and min_coordinate <= coordinate:
                push(node.left)
            if node.right is not _NIL and coordinate <= max_coordinate:
                push(node.right)
