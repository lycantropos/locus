from heapq import (heappush,
                   heapreplace)
from typing import (Iterator,
                    List,
                    Sequence,
                    Tuple,
                    Type,
                    Union)

from reprit.base import generate_repr

from .core import interval as _interval
from .core.utils import (linear_distance as _linear_distance,
                         planar_distance as _planar_distance)
from .hints import (Coordinate,
                    Interval,
                    Point)

Item = Tuple[int, Point]
NIL = None


class Node:
    """
    Represents node of *kd*-tree.

    Can be subclassed for custom metrics definition.
    """

    __slots__ = 'index', 'point', 'axis', 'left', 'right'

    def __init__(self,
                 index: int,
                 point: Point,
                 axis: int,
                 left: Union['Node', NIL],
                 right: Union['Node', NIL]) -> None:
        """
        Initializes node.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> node = Node(5, (-10, 10), 0, NIL, NIL)
        """
        self.index = index
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right

    @property
    def coordinate(self) -> Coordinate:
        """
        Returns coordinate on which branching was performed.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> node = Node(5, (-10, 10), 0, NIL, NIL)
        >>> node.coordinate == -10
        True
        """
        return self.point[self.axis]

    @property
    def item(self) -> Item:
        """
        Returns underlying index with point.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> node = Node(5, (-10, 10), 0, NIL, NIL)
        >>> node.item == (5, (-10, 10))
        True
        """
        return self.index, self.point

    def distance_to_point(self, point: Point) -> Coordinate:
        """
        Calculates distance to given point.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> node = Node(5, (-10, 10), 0, NIL, NIL)
        >>> node.distance_to_point((-7, 6)) == 5
        True
        """
        return _planar_distance(self.point, point)

    def distance_to_coordinate(self, coordinate: Coordinate) -> Coordinate:
        """
        Calculates distance to given coordinate.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> node = Node(5, (-10, 10), 0, NIL, NIL)
        >>> node.distance_to_coordinate(-1) == 9
        True
        """
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

        >>> points = list(zip(range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.n_nearest_points(2, (0, 0)) == [(-3, 2), (-2, 3)]
        True
        >>> tree.n_nearest_points(len(points), (0, 0)) == points
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

        >>> points = list(zip(range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.n_nearest_items(2, (0, 0)) == [(2, (-3, 2)), (3, (-2, 3))]
        True
        >>> (tree.n_nearest_items(len(points), (0, 0))
        ...  == list(enumerate(points)))
        True
        """
        return (self._n_nearest_items(n, point)
                if n < len(self._points)
                else list(enumerate(self._points)))

    def _n_nearest_items(self, n: int, point: Point) -> List[Item]:
        candidates = []
        queue = [self._root]
        push, pop = queue.append, queue.pop
        while queue:
            node = pop()
            distance_to_point = node.distance_to_point(point)
            candidate = -distance_to_point, node.item
            if len(candidates) < n:
                heappush(candidates, candidate)
            elif distance_to_point < -candidates[0][0]:
                heapreplace(candidates, candidate)
            point_is_on_the_left = point[node.axis] < node.coordinate
            if point_is_on_the_left:
                if node.left is not NIL:
                    push(node.left)
            elif node.right is not NIL:
                push(node.right)
            if (len(candidates) < n
                    or (node.distance_to_coordinate(point[node.axis])
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

        >>> points = list(zip(range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.nearest_index((0, 0)) == 2
        True
        >>> tree.nearest_index((-3, 2)) == 2
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

        >>> points = list(zip(range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.nearest_point((0, 0)) == (-3, 2)
        True
        >>> tree.nearest_point((-3, 2)) == (-3, 2)
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

        >>> points = list(zip(range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.nearest_item((0, 0)) == (2, (-3, 2))
        True
        >>> tree.nearest_item((-3, 2)) == (2, (-3, 2))
        True
        """
        node = self._root
        result, min_distance = node.item, node.distance_to_point(point)
        queue = [node]
        push, pop = queue.append, queue.pop
        while queue:
            node = pop()
            distance_to_point = node.distance_to_point(point)
            if distance_to_point < min_distance:
                result, min_distance = node.item, distance_to_point
            point_is_on_the_left = point[node.axis] < node.coordinate
            if point_is_on_the_left:
                if node.left is not NIL:
                    push(node.left)
            elif node.right is not NIL:
                push(node.right)
            if node.distance_to_coordinate(point[node.axis]) < min_distance:
                if point_is_on_the_left:
                    if node.right is not NIL:
                        push(node.right)
                elif node.left is not NIL:
                    push(node.left)
        return result

    def find_ball_indices(self, center: Point, radius: Coordinate
                          ) -> List[int]:
        """
        Searches for indices of points in the tree
        that lie inside the closed ball with given center and radius.

        Time complexity:
            ``O(dimension * size ** (1 - 1 / dimension) + hits_count)``
        Memory complexity:
            ``O(dimension * size ** (1 - 1 / dimension) + hits_count)``

        where ``dimension = len(self.points[0])``, ``size = len(self.points)``,
        ``hits_count`` --- number of found indices.

        Reference:
            https://en.wikipedia.org/wiki/K-d_tree#Range_search

        :param center: center of the ball.
        :param radius: radius of the ball.
        :returns: indices of points in the tree that lie inside the ball.

        >>> points = list(zip(range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.find_ball_indices((0, 0), 0) == []
        True
        >>> tree.find_ball_indices((0, 0), 2) == []
        True
        >>> tree.find_ball_indices((0, 0), 4) == [2, 3]
        True
        """
        return [index for index, _ in self._find_ball_items(center, radius)]

    def find_ball_points(self, center: Point, radius: Coordinate
                         ) -> List[Point]:
        """
        Searches for points in the tree
        that lie inside the closed ball with given center and radius.

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
        :returns: points in the tree that lie inside the ball.

        >>> points = list(zip(range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.find_ball_points((0, 0), 0) == []
        True
        >>> tree.find_ball_points((0, 0), 2) == []
        True
        >>> tree.find_ball_points((0, 0), 4) == [(-3, 2), (-2, 3)]
        True
        """
        return [point for _, point in self._find_ball_items(center, radius)]

    def find_ball_items(self, center: Point, radius: Coordinate) -> List[Item]:
        """
        Searches for indices with points in the tree
        that lie inside the closed ball with given center and radius.

        Time complexity:
            ``O(dimension * size ** (1 - 1 / dimension) + hits_count)``
        Memory complexity:
            ``O(dimension * size ** (1 - 1 / dimension) + hits_count)``

        where ``dimension = len(self.points[0])``, ``size = len(self.points)``,
        ``hits_count`` --- number of found indices with points.

        Reference:
            https://en.wikipedia.org/wiki/K-d_tree#Range_search

        :param center: center of the ball.
        :param radius: radius of the ball.
        :returns: indices with points in the tree that lie inside the ball.

        >>> points = list(zip(range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.find_ball_items((0, 0), 0) == []
        True
        >>> tree.find_ball_items((0, 0), 2) == []
        True
        >>> tree.find_ball_items((0, 0), 4) == [(2, (-3, 2)), (3, (-2, 3))]
        True
        """
        return list(self._find_ball_items(center, radius))

    def _find_ball_items(self, center: Point, radius: Coordinate
                         ) -> Iterator[Item]:
        queue = [self._root]
        push, pop = queue.append, queue.pop
        while queue:
            node = pop()
            if node.distance_to_point(center) <= radius:
                yield node.item
            hyperplane_delta = center[node.axis] - node.coordinate
            if node.left is not NIL and hyperplane_delta <= radius:
                push(node.left)
            if node.right is not NIL and -radius <= hyperplane_delta:
                push(node.right)

    def find_interval_indices(self, interval: Interval) -> List[int]:
        """
        Searches for indices of points that lie inside the closed interval.

        Time complexity:
            ``O(dimension * size ** (1 - 1 / dimension) + hits_count)``
        Memory complexity:
            ``O(dimension * size ** (1 - 1 / dimension) + hits_count)``

        where ``dimension = len(self.points[0])``, ``size = len(self.points)``,
        ``hits_count`` --- number of found indices.

        Reference:
            https://en.wikipedia.org/wiki/K-d_tree#Range_search

        :param interval: interval to search in.
        :returns: indices of points that lie inside the interval.

        >>> points = list(zip(range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.find_interval_indices(((-3, 3), (0, 1))) == []
        True
        >>> tree.find_interval_indices(((-3, 3), (0, 2))) == [2]
        True
        >>> tree.find_interval_indices(((-3, 3), (0, 3))) == [2, 3]
        True
        """
        return [index for index, _ in self._find_interval_items(interval)]

    def find_interval_points(self, interval: Interval) -> List[Point]:
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
        :returns: points that lie inside the interval.

        >>> points = list(zip(range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.find_interval_points(((-3, 3), (0, 1))) == []
        True
        >>> tree.find_interval_points(((-3, 3), (0, 2))) == [(-3, 2)]
        True
        >>> tree.find_interval_points(((-3, 3), (0, 3))) == [(-3, 2), (-2, 3)]
        True
        """
        return [point for _, point in self._find_interval_items(interval)]

    def find_interval_items(self, interval: Interval) -> List[Item]:
        """
        Searches for indices with points in the tree
        that lie inside the closed interval.

        Time complexity:
            ``O(dimension * size ** (1 - 1 / dimension) + hits_count)``
        Memory complexity:
            ``O(dimension * size ** (1 - 1 / dimension) + hits_count)``

        where ``dimension = len(self.points[0])``, ``size = len(self.points)``,
        ``hits_count`` --- number of found indices with points.

        Reference:
            https://en.wikipedia.org/wiki/K-d_tree#Range_search

        :param interval: interval to search in.
        :returns:
            indices with points in the tree that lie inside the interval.

        >>> points = list(zip(range(-5, 6), range(10)))
        >>> tree = Tree(points)
        >>> tree.find_interval_items(((-3, 3), (0, 1))) == []
        True
        >>> tree.find_interval_items(((-3, 3), (0, 2))) == [(2, (-3, 2))]
        True
        >>> (tree.find_interval_items(((-3, 3), (0, 3)))
        ...  == [(2, (-3, 2)), (3, (-2, 3))])
        True
        """
        return list(self._find_interval_items(interval))

    def _find_interval_items(self, interval: Interval) -> List[Item]:
        queue = [self._root]
        push, pop = queue.append, queue.pop
        while queue:
            node = pop()
            if _interval.contains_point(interval, node.point):
                yield node.item
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
    middle_index = (len(indices) - 1) // 2
    pivot_index = indices[middle_index]
    next_axis = (axis + 1) % dimension
    return cls(pivot_index, points[pivot_index], axis,
               _create_node(cls, indices[:middle_index], points, dimension,
                            next_axis),
               _create_node(cls, indices[middle_index + 1:], points, dimension,
                            next_axis))
