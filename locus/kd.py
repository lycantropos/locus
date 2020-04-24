from heapq import (heappush,
                   heapreplace)
from operator import itemgetter
from typing import (Iterable,
                    Iterator,
                    List,
                    Union)

from reprit.base import generate_repr

from .core.utils import (point_in_interval as _point_in_interval,
                         squared_distance as _squared_distance)
from .hints import (Coordinate,
                    Interval,
                    Point)

NIL = None


class Node:
    __slots__ = 'point', 'axis', 'left', 'right'

    def __init__(self,
                 point: Point,
                 axis: int,
                 left: Union['Node', NIL],
                 right: Union['Node', NIL]) -> None:
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right

    __repr__ = generate_repr(__init__)


class Tree:
    __slots__ = 'root',

    def __init__(self, root: Union[Node, NIL]) -> None:
        self.root = root

    def __iter__(self) -> Iterator[Point]:
        node = self.root
        queue = []
        push, pop = queue.append, queue.pop
        while True:
            while node is not NIL:
                push(node)
                node = node.left
            if not queue:
                return
            node = pop()
            yield node.point
            node = node.right

    @classmethod
    def from_iterable(cls, _points: Iterable[Point]) -> 'Tree':
        points = list(_points)
        dimension = len(points[0])
        return cls(_create_node(points, 0, dimension))

    __repr__ = generate_repr(from_iterable)

    @property
    def _points(self) -> List[Point]:
        return list(self)

    def query_ball(self, center: Point, radius: Coordinate) -> List[Point]:
        return list(self._query_ball(center, radius))

    def _query_ball(self, center: Point,
                    radius: Coordinate) -> Iterator[Point]:
        queue = [self.root]
        push, pop = queue.append, queue.pop
        squared_radius = radius * radius
        while queue:
            node = pop()
            if _squared_distance(node.point, center) <= squared_radius:
                yield node.point
            hyperplane_delta = center[node.axis] - node.point[node.axis]
            if node.left is not NIL and hyperplane_delta <= radius:
                push(node.left)
            if node.right is not NIL and -radius <= hyperplane_delta:
                push(node.right)

    def query_interval(self, interval: Interval) -> List[Point]:
        return list(self._query_interval(interval))

    def _query_interval(self, interval: Interval) -> List[Point]:
        queue = [self.root]
        push, pop = queue.append, queue.pop
        while queue:
            node = pop()
            if _point_in_interval(node.point, interval):
                yield node.point
            min_coordinate, max_coordinate = interval[node.axis]
            hyperplane = node.point[node.axis]
            if node.left is not NIL and min_coordinate <= hyperplane:
                push(node.left)
            if node.right is not NIL and hyperplane <= max_coordinate:
                push(node.right)

    def n_nearest(self, n: int, point: Point) -> List[Point]:
        items = []
        queue = [self.root]
        push, pop = queue.append, queue.pop
        while queue:
            node = pop()
            node_point = node.point
            distance_to_point = _squared_distance(node_point, point)
            item = -distance_to_point, node_point
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
        return items

    def nearest(self, point: Point) -> Point:
        result, = self.n_nearest(1, point)
        return result


def _create_node(points: List[Point],
                 axis: int,
                 dimension: int) -> Union[Node, NIL]:
    if not points:
        return NIL
    middle_index = len(points) // 2
    points = sorted(points,
                    key=itemgetter(axis))
    next_axis = (axis + 1) % dimension
    return Node(points[middle_index], axis,
                _create_node(points[:middle_index], next_axis, dimension),
                _create_node(points[middle_index + 1:], next_axis, dimension))


def tree(points: Iterable[Point]) -> Tree:
    return Tree.from_iterable(points)
