from heapq import (heappush,
                   heapreplace)
from typing import (Iterator,
                    List,
                    Sequence,
                    Tuple,
                    Type,
                    Union)

from reprit.base import generate_repr

from .core.utils import (linear_distance as _linear_distance,
                         planar_distance as _planar_distance,
                         point_in_interval as _point_in_interval)
from .hints import (Coordinate,
                    Interval,
                    Point)

Item = Tuple[int, Point]
NIL = None


class Node:
    __slots__ = 'index', 'point', 'axis', 'left', 'right'

    def __init__(self,
                 index: int,
                 point: Point,
                 axis: int,
                 left: Union['Node', NIL],
                 right: Union['Node', NIL]) -> None:
        self.index = index
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right

    @property
    def coordinate(self) -> Coordinate:
        return self.point[self.axis]

    def distance_to_point(self, point: Point) -> Coordinate:
        return _planar_distance(self.point, point)

    def distance_to_coordinate(self, coordinate: Coordinate) -> Coordinate:
        return _linear_distance(self.coordinate, coordinate)

    __repr__ = generate_repr(__init__)


class Tree:
    """
    Represents `k`-dimensional (aka *kd*) tree.

    Reference:
        https://en.wikipedia.org/wiki/K-d_tree
    """

    __slots__ = '_points', '_root'

    def __init__(self, points: Sequence[Point],
                 *,
                 node_cls: Type[Node] = Node) -> None:
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
        self._root = _create_node(node_cls, range(len(points)), points,
                                  len(points[0]), 0)

    __repr__ = generate_repr(__init__)

    @property
    def node_cls(self) -> Type[Node]:
        """
        Returns type of the nodes.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> points = list(zip(range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.node_cls is Node
        True
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
        return ([point for _, point in self._n_nearest_items(n, point)]
                if n < len(self._points)
                else self._points)

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
        return ([index for index, _ in self._n_nearest_items(n, point)]
                if n < len(self._points)
                else range(len(self._points)))

    def _n_nearest_items(self, n: int, point: Point) -> List[Item]:
        items = []
        queue = [self._root]
        push, pop = queue.append, queue.pop
        while queue:
            node = pop()
            distance_to_point = node.distance_to_point(point)
            item = -distance_to_point, node.index, node.point
            if len(items) < n:
                heappush(items, item)
            elif distance_to_point < -items[0][0]:
                heapreplace(items, item)
            point_is_on_the_left = point[node.axis] < node.coordinate
            if point_is_on_the_left:
                if node.left is not NIL:
                    push(node.left)
            elif node.right is not NIL:
                push(node.right)
            if len(items) < n or (node.distance_to_coordinate(point[node.axis])
                                  < -items[0][0]):
                if point_is_on_the_left:
                    if node.right is not NIL:
                        push(node.right)
                elif node.left is not NIL:
                    push(node.left)
        return [(index, point) for _, index, point in items]

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
        return [point for _, point in self._query_ball_items(center, radius)]

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
        return [index for index, _ in self._query_ball_items(center, radius)]

    def _query_ball_items(self, center: Point, radius: Coordinate
                          ) -> Iterator[Item]:
        queue = [self._root]
        push, pop = queue.append, queue.pop
        while queue:
            node = pop()
            if node.distance_to_point(center) <= radius:
                yield node.index, node.point
            hyperplane_delta = center[node.axis] - node.coordinate
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
        return [point for _, point in self._query_interval_items(interval)]

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
        return [index for index, _ in self._query_interval_items(interval)]

    def _query_interval_items(self, interval: Interval) -> List[Item]:
        queue = [self._root]
        push, pop = queue.append, queue.pop
        while queue:
            node = pop()
            if _point_in_interval(node.point, interval):
                yield node.index, node.point
            min_coordinate, max_coordinate = interval[node.axis]
            if node.left is not NIL and min_coordinate <= node.coordinate:
                push(node.left)
            if node.right is not NIL and node.coordinate <= max_coordinate:
                push(node.right)


def _create_node(cls: Type[Node],
                 indices: Sequence[int],
                 points: Sequence[Point],
                 dimension: int,
                 axis: int) -> Union[Node, NIL]:
    if not indices:
        return NIL
    indices = sorted(indices,
                     key=lambda index: points[index][axis])
    middle_index = len(indices) // 2
    pivot_index = indices[middle_index]
    next_axis = (axis + 1) % dimension
    return cls(pivot_index, points[pivot_index], axis,
               _create_node(cls, indices[:middle_index], points, dimension,
                            next_axis),
               _create_node(cls, indices[middle_index + 1:], points, dimension,
                            next_axis))
