from heapq import (heappush,
                   heapreplace)
from operator import attrgetter
from typing import (Iterator,
                    List,
                    Optional,
                    Sequence,
                    Tuple,
                    Type,
                    Union)

from ground.hints import (Box,
                          Coordinate,
                          Point)
from reprit.base import generate_repr

from .core import box as _box
from .core.utils import points_distance as _points_distance

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

Item = Tuple[int, Point]
NIL = None

_PROJECTORS = attrgetter('x'), attrgetter('y')


class Node:
    """
    Represents node of *kd*-tree.
    """

    __slots__ = 'index', 'point', 'is_y_axis', 'projector', 'left', 'right'

    def __init__(self,
                 index: int,
                 point: Point,
                 is_y_axis: bool,
                 left: Union['Node', NIL],
                 right: Union['Node', NIL]) -> None:
        self.index, self.point = index, point
        self.is_y_axis, self.projector = is_y_axis, _PROJECTORS[is_y_axis]
        self.left, self.right = left, right

    __repr__ = generate_repr(__init__)

    @property
    def item(self) -> Item:
        """Returns item of the node."""
        return self.index, self.point

    @property
    def projection(self) -> Coordinate:
        """Returns projection of the node point onto the corresponding axis."""
        return self.projector(self.point)

    def distance_to_point(self, point: Point) -> Coordinate:
        """Calculates distance to given point."""
        return _points_distance(self.point, point)

    def distance_to_coordinate(self, coordinate: Coordinate) -> Coordinate:
        """Calculates distance to given coordinate."""
        return (self.projection - coordinate) ** 2


class Tree:
    """
    Represents `k`-dimensional (aka *kd*) tree.

    Reference:
        https://en.wikipedia.org/wiki/K-d_tree
    """

    __slots__ = '_points', '_root'

    def __init__(self, points: Sequence[Point],
                 *,
                 node_cls: Optional[Type[Node]] = None) -> None:
        """
        Initializes tree from points.

        Time complexity:
            ``O(dimension * size * log size)``
        Memory complexity:
            ``O(dimension * size)``

        where ``dimension = len(points[0])``, ``size = len(points)``.

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point = context.point_cls
        >>> points = list(map(Point, range(-5, 6), range(10)))
        >>> tree = Tree(points)
        """
        self._points = points
        self._root = _create_node(Node if node_cls is None else node_cls,
                                  range(len(points)), points, False)

    __repr__ = generate_repr(__init__)

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
    def points(self) -> Sequence[Point]:
        """
        Returns underlying points.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point = context.point_cls
        >>> points = list(map(Point, range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.points == points
        True
        """
        return self._points

    def n_nearest_indices(self, n: int, point: Point) -> Sequence[int]:
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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point = context.point_cls
        >>> points = list(map(Point, range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.n_nearest_indices(2, Point(0, 0)) == [2, 3]
        True
        >>> (tree.n_nearest_indices(len(points), Point(0, 0))
        ...  == range(len(points)))
        True
        """
        return ([index for index, _ in self._n_nearest_items(n, point)]
                if n < len(self._points)
                else range(len(self._points)))

    def n_nearest_points(self, n: int, point: Point) -> Sequence[Point]:
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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point = context.point_cls
        >>> points = list(map(Point, range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> (tree.n_nearest_points(2, Point(0, 0))
        ...  == [Point(-3, 2), Point(-2, 3)])
        True
        >>> tree.n_nearest_points(len(points), Point(0, 0)) == points
        True
        """
        return ([point for _, point in self._n_nearest_items(n, point)]
                if n < len(self._points)
                else self._points)

    def n_nearest_items(self, n: int, point: Point) -> Sequence[Item]:
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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point = context.point_cls
        >>> points = list(map(Point, range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> (tree.n_nearest_items(2, Point(0, 0))
        ...  == [(2, Point(-3, 2)), (3, Point(-2, 3))])
        True
        >>> (tree.n_nearest_items(len(points), Point(0, 0))
        ...  == list(enumerate(points)))
        True
        """
        return (self._n_nearest_items(n, point)
                if n < len(self._points)
                else list(enumerate(self._points)))

    def _n_nearest_items(self, n: int, point: Point) -> List[Item]:
        candidates = []  # type: List[Tuple[Coordinate, Item]]
        queue = [self._root]
        push, pop = queue.append, queue.pop
        while queue:
            node = pop()  # type: Node
            distance_to_point = node.distance_to_point(point)
            candidate = -distance_to_point, node.item
            if len(candidates) < n:
                heappush(candidates, candidate)
            elif distance_to_point < -candidates[0][0]:
                heapreplace(candidates, candidate)
            coordinate = node.projector(point)
            point_is_on_the_left = coordinate < node.projection
            if point_is_on_the_left:
                if node.left is not NIL:
                    push(node.left)
            elif node.right is not NIL:
                push(node.right)
            if (len(candidates) < n
                    or (node.distance_to_coordinate(coordinate)
                        < -candidates[0][0])):
                if point_is_on_the_left:
                    if node.right is not NIL:
                        push(node.right)
                elif node.left is not NIL:
                    push(node.left)
        return [item for _, item in candidates]

    def nearest_index(self, point: Point) -> int:
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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point = context.point_cls
        >>> points = list(map(Point, range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.nearest_index(Point(0, 0)) == 2
        True
        >>> tree.nearest_index(Point(-3, 2)) == 2
        True
        """
        result, _ = self.nearest_item(point)
        return result

    def nearest_point(self, point: Point) -> Point:
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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point = context.point_cls
        >>> points = list(map(Point, range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.nearest_point(Point(0, 0)) == Point(-3, 2)
        True
        >>> tree.nearest_point(Point(-3, 2)) == Point(-3, 2)
        True
        """
        _, result = self.nearest_item(point)
        return result

    def nearest_item(self, point: Point) -> Item:
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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Point = context.point_cls
        >>> points = list(map(Point, range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.nearest_item(Point(0, 0)) == (2, Point(-3, 2))
        True
        >>> tree.nearest_item(Point(-3, 2)) == (2, Point(-3, 2))
        True
        """
        node = self._root
        result, min_distance = node.item, node.distance_to_point(point)
        queue = [node]
        push, pop = queue.append, queue.pop
        while queue:
            node = pop()  # type: Node
            distance_to_point = node.distance_to_point(point)
            if distance_to_point < min_distance:
                result, min_distance = node.item, distance_to_point
            coordinate = node.projector(point)
            point_is_on_the_left = coordinate < node.projection
            if point_is_on_the_left:
                if node.left is not NIL:
                    push(node.left)
            elif node.right is not NIL:
                push(node.right)
            if node.distance_to_coordinate(coordinate) < min_distance:
                if point_is_on_the_left:
                    if node.right is not NIL:
                        push(node.right)
                elif node.left is not NIL:
                    push(node.left)
        return result

    def find_box_indices(self, box: Box) -> List[int]:
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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Box, Point = context.box_cls, context.point_cls
        >>> points = list(map(Point, range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.find_box_indices(Box(-3, 3, 0, 1)) == []
        True
        >>> tree.find_box_indices(Box(-3, 3, 0, 2)) == [2]
        True
        >>> tree.find_box_indices(Box(-3, 3, 0, 3)) == [2, 3]
        True
        """
        return [index for index, _ in self._find_box_items(box)]

    def find_box_points(self, box: Box) -> List[Point]:
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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Box, Point = context.box_cls, context.point_cls
        >>> points = list(map(Point, range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.find_box_points(Box(-3, 3, 0, 1)) == []
        True
        >>> tree.find_box_points(Box(-3, 3, 0, 2)) == [Point(-3, 2)]
        True
        >>> (tree.find_box_points(Box(-3, 3, 0, 3))
        ...  == [Point(-3, 2), Point(-2, 3)])
        True
        """
        return [point for _, point in self._find_box_items(box)]

    def find_box_items(self, box: Box) -> List[Item]:
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

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Box, Point = context.box_cls, context.point_cls
        >>> points = list(map(Point, range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.find_box_items(Box(-3, 3, 0, 1)) == []
        True
        >>> tree.find_box_items(Box(-3, 3, 0, 2)) == [(2, Point(-3, 2))]
        True
        >>> (tree.find_box_items(Box(-3, 3, 0, 3))
        ...  == [(2, Point(-3, 2)), (3, Point(-2, 3))])
        True
        """
        return list(self._find_box_items(box))

    def _find_box_items(self, box: Box) -> Iterator[Item]:
        queue = [self._root]
        push, pop = queue.append, queue.pop
        while queue:
            node = pop()  # type: Node
            if _box.contains_point(box, node.point):
                yield node.item
            min_coordinate, max_coordinate = ((box.min_y, box.max_y)
                                              if node.is_y_axis
                                              else (box.min_x, box.max_x))
            coordinate = node.projector(node.point)
            if node.left is not NIL and min_coordinate <= coordinate:
                push(node.left)
            if node.right is not NIL and coordinate <= max_coordinate:
                push(node.right)


def _create_node(cls: Type[Node],
                 indices: Sequence[int],
                 points: Sequence[Point],
                 is_y_axis: bool) -> Union[Node, NIL]:
    if not indices:
        return NIL
    indices = sorted(indices,
                     key=(lambda index, projector=_PROJECTORS[is_y_axis]
                          : projector(points[index])))
    middle_index = (len(indices) - 1) // 2
    pivot_index = indices[middle_index]
    next_is_y_axis = not is_y_axis
    return cls(pivot_index, points[pivot_index], is_y_axis,
               _create_node(cls, indices[:middle_index], points,
                            next_is_y_axis),
               _create_node(cls, indices[middle_index + 1:], points,
                            next_is_y_axis))
