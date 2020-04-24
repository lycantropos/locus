from heapq import (heappush,
                   heapreplace)
from typing import (Iterator,
                    List,
                    Sequence,
                    Union)

from reprit.base import generate_repr

from .core.utils import (point_in_interval as _point_in_interval,
                         squared_distance as _squared_distance)
from .hints import (Coordinate,
                    Interval,
                    Point)

NIL = None


class Node:
    __slots__ = 'index', 'axis', 'left', 'right'

    def __init__(self,
                 index: int,
                 axis: int,
                 left: Union['Node', NIL],
                 right: Union['Node', NIL]) -> None:
        self.index = index
        self.axis = axis
        self.left = left
        self.right = right

    __repr__ = generate_repr(__init__)


class Tree:
    """
    Represents `k`-dimensional (aka *kd*) tree.

    Reference:
        https://en.wikipedia.org/wiki/K-d_tree
    """

    __slots__ = '_root', '_points'

    def __init__(self, points: Sequence[Point]) -> None:
        """
        Initializes tree from points.

        Time complexity:
            ``O(dimension * size * log size)``
        Memory complexity:
            ``O(dimension * size)``

        where ``dimension = len(points[0])``, ``size = len(points)``.

        >>> points = list(zip(range(-5, 6), range(10)))
        >>> tree = Tree(points)
        """
        self._points = points
        self._root = _create_node(points, range(len(points)), len(points[0]),
                                  0)

    __repr__ = generate_repr(__init__)

    @property
    def points(self) -> Sequence[Point]:
        """
        Returns underlying points.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> points = list(zip(range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.points == points
        True
        """
        return self._points

    def n_nearest(self, n: int, point: Point) -> Sequence[Point]:
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

        >>> points = list(zip(range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.n_nearest(2, (0, 0)) == [(-3, 2), (-2, 3)]
        True
        >>> set(tree.n_nearest(len(points), (0, 0))) == set(points)
        True
        """
        return (self._points
                if n >= len(self._points)
                else [self._points[index]
                      for index in self.n_nearest_indices(n, point)])

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

        >>> points = list(zip(range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.n_nearest_indices(2, (0, 0)) == [2, 3]
        True
        >>> tree.n_nearest_indices(len(points), (0, 0)) == range(len(points))
        True
        """
        if n >= len(self._points):
            return range(len(self._points))
        items = []
        points, queue = self._points, [self._root]
        push, pop = queue.append, queue.pop
        while queue:
            node = pop()
            node_point = points[node.index]
            distance_to_point = _squared_distance(node_point, point)
            item = -distance_to_point, node.index
            if len(items) < n:
                heappush(items, item)
            elif distance_to_point < -items[0][0]:
                heapreplace(items, item)
            hyperplane_delta = point[node.axis] - node_point[node.axis]
            point_is_on_the_left = hyperplane_delta < 0
            if point_is_on_the_left:
                if node.left is not NIL:
                    push(node.left)
            elif node.right is not NIL:
                push(node.right)
            if len(items) < n or hyperplane_delta ** 2 < -items[0][0]:
                if point_is_on_the_left:
                    if node.right is not NIL:
                        push(node.right)
                elif node.left is not NIL:
                    push(node.left)
        return [index for _, index in items]

    def nearest(self, point: Point) -> Point:
        """
        Searches for point in the tree that is nearest to the given point.

        Time complexity:
            ``O(log size)``
        Memory complexity:
            ``O(log size)``

        where ``size = len(self.points)``.

        Reference:
            https://en.wikipedia.org/wiki/K-d_tree#Nearest_neighbour_search

        :param point: input point.
        :returns: point in the tree nearest to the input point.

        >>> points = list(zip(range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.nearest((0, 0)) == (-3, 2)
        True
        >>> tree.nearest((-3, 2)) == (-3, 2)
        True
        """
        result, = self.n_nearest(1, point)
        return result

    def nearest_index(self, point: Point) -> int:
        """
        Searches for index of a point in the tree
        that is nearest to the given point.

        Time complexity:
            ``O(log size)``
        Memory complexity:
            ``O(log size)``

        where ``size = len(self.points)``.

        Reference:
            https://en.wikipedia.org/wiki/K-d_tree#Nearest_neighbour_search

        :param point: input point.
        :returns: index of a point in the tree nearest to the input point.

        >>> points = list(zip(range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.nearest_index((0, 0)) == 2
        True
        >>> tree.nearest_index((-3, 2)) == 2
        True
        """
        result, = self.n_nearest_indices(1, point)
        return result

    def query_ball(self, center: Point, radius: Coordinate) -> List[Point]:
        """
        Searches for points that lie inside the closed ball
        with given center and radius.

        Time complexity:
            ``O(dimension * size ** (1 - 1 / dimension) + hits_count)``
        Memory complexity:
            ``O(dimension * size ** (1 - 1 / dimension) + hits_count)``

        where ``dimension = len(self.points[0])``, ``size = len(self.points)``,
        ``hits_count`` --- number of found points.

        Reference:
            https://en.wikipedia.org/wiki/K-d_tree#Range_search

        :param center: center of the ball.
        :param radius: radius of the ball.
        :returns: points which lie inside the ball.

        >>> points = list(zip(range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.query_ball((0, 0), 0) == []
        True
        >>> tree.query_ball((0, 0), 2) == []
        True
        >>> tree.query_ball((0, 0), 4) == [(-3, 2), (-2, 3)]
        True
        """
        return [self.points[index]
                for index in self._query_ball_indices(center, radius)]

    def query_ball_indices(self, center: Point,
                           radius: Coordinate) -> List[int]:
        """
        Searches for indices of points that lie inside the closed ball
        with given center and radius.

        Time complexity:
            ``O(dimension * size ** (1 - 1 / dimension) + hits_count)``
        Memory complexity:
            ``O(dimension * size ** (1 - 1 / dimension) + hits_count)``

        where ``dimension = len(self.points[0])``, ``size = len(self.points)``,
        ``hits_count`` --- number of found points.

        Reference:
            https://en.wikipedia.org/wiki/K-d_tree#Range_search

        :param center: center of the ball.
        :param radius: radius of the ball.
        :returns: indices of points which lie inside the ball.

        >>> points = list(zip(range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.query_ball_indices((0, 0), 0) == []
        True
        >>> tree.query_ball_indices((0, 0), 2) == []
        True
        >>> tree.query_ball_indices((0, 0), 4) == [2, 3]
        True
        """
        return list(self._query_ball_indices(center, radius))

    def _query_ball_indices(self, center: Point,
                            radius: Coordinate) -> Iterator[int]:
        points, queue = self.points, [self._root]
        push, pop = queue.append, queue.pop
        squared_radius = radius * radius
        while queue:
            node = pop()
            node_point = points[node.index]
            if _squared_distance(node_point, center) <= squared_radius:
                yield node.index
            hyperplane_delta = center[node.axis] - node_point[node.axis]
            if node.left is not NIL and hyperplane_delta <= radius:
                push(node.left)
            if node.right is not NIL and -radius <= hyperplane_delta:
                push(node.right)

    def query_interval(self, interval: Interval) -> List[Point]:
        """
        Searches for points that lie inside the closed interval.

        Time complexity:
            ``O(dimension * size ** (1 - 1 / dimension) + hits_count)``
        Memory complexity:
            ``O(dimension * size ** (1 - 1 / dimension) + hits_count)``

        where ``dimension = len(self.points[0])``, ``size = len(self.points)``,
        ``hits_count`` --- number of found points.

        Reference:
            https://en.wikipedia.org/wiki/K-d_tree#Range_search

        :param interval: interval to search in.
        :returns: points which lie inside the interval.

        >>> points = list(zip(range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.query_interval(((-3, 3), (0, 1))) == []
        True
        >>> tree.query_interval(((-3, 3), (0, 2))) == [(-3, 2)]
        True
        >>> tree.query_interval(((-3, 3), (0, 3))) == [(-3, 2), (-2, 3)]
        True
        """
        return [self.points[index]
                for index in self._query_interval_indices(interval)]

    def query_interval_indices(self, interval: Interval) -> List[int]:
        """
        Searches for indices of points that lie inside the closed interval.

        Time complexity:
            ``O(dimension * size ** (1 - 1 / dimension) + hits_count)``
        Memory complexity:
            ``O(dimension * size ** (1 - 1 / dimension) + hits_count)``

        where ``dimension = len(self.points[0])``, ``size = len(self.points)``,
        ``hits_count`` --- number of found points.

        Reference:
            https://en.wikipedia.org/wiki/K-d_tree#Range_search

        :param interval: interval to search in.
        :returns: indices of points which lie inside the interval.

        >>> points = list(zip(range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.query_interval_indices(((-3, 3), (0, 1))) == []
        True
        >>> tree.query_interval_indices(((-3, 3), (0, 2))) == [2]
        True
        >>> tree.query_interval_indices(((-3, 3), (0, 3))) == [2, 3]
        True
        """
        return list(self._query_interval_indices(interval))

    def _query_interval_indices(self, interval: Interval) -> List[int]:
        points, queue = self.points, [self._root]
        push, pop = queue.append, queue.pop
        while queue:
            node = pop()
            node_point = points[node.index]
            if _point_in_interval(node_point, interval):
                yield node.index
            min_coordinate, max_coordinate = interval[node.axis]
            hyperplane = node_point[node.axis]
            if node.left is not NIL and min_coordinate <= hyperplane:
                push(node.left)
            if node.right is not NIL and hyperplane <= max_coordinate:
                push(node.right)


def _create_node(points: Sequence[Point],
                 indices: Sequence[int],
                 dimension: int,
                 axis: int) -> Union[Node, NIL]:
    if not indices:
        return NIL
    pivot_index = len(indices) // 2
    indices = sorted(indices,
                     key=lambda index: points[index][axis])
    next_axis = (axis + 1) % dimension
    return Node(indices[pivot_index], axis,
                _create_node(points, indices[:pivot_index], dimension,
                             next_axis),
                _create_node(points, indices[pivot_index + 1:], dimension,
                             next_axis))
