from functools import reduce as _reduce
from heapq import (heappop as _heappop,
                   heappush as _heappush)
from math import floor as _floor
from typing import (Callable as _Callable,
                    Iterator as _Iterator,
                    List as _List,
                    Optional as _Optional,
                    Sequence as _Sequence,
                    Tuple as _Tuple)

from ground.base import (Context as _Context,
                         get_context as _get_context)
from ground.hints import (Box as _Box,
                          Coordinate as _Coordinate,
                          Point as _Point)
from reprit.base import generate_repr as _generate_repr

from .core import (box as _box,
                   hilbert as _hilbert)
from .core.utils import ceil_division as _ceil_division

Item = _Tuple[int, _Box]


class Node:
    """Represents node of *R*-tree."""
    __slots__ = 'box', 'children', 'index', 'metric'

    def __init__(self,
                 index: int,
                 box: _Box,
                 children: _Optional[_Sequence['Node']],
                 metric: _Callable[[_Box, _Point], _Coordinate]) -> None:
        self.box, self.children, self.index, self.metric = (box, children,
                                                            index, metric)

    __repr__ = _generate_repr(__init__)

    @property
    def is_leaf(self) -> bool:
        """Checks whether the node is a leaf."""
        return self.children is None

    @property
    def item(self) -> Item:
        """Returns underlying index with box."""
        return self.index, self.box

    def distance_to_point(self, point: _Point) -> _Coordinate:
        """Calculates distance to given point."""
        return self.metric(self.box, point)


class Tree:
    """
    Represents packed 2-dimensional Hilbert *R*-tree.

    Reference:
        https://en.wikipedia.org/wiki/Hilbert_R-tree#Packed_Hilbert_R-trees
    """
    __slots__ = '_boxes', '_context', '_max_children', '_root'

    def __init__(self,
                 boxes: _Sequence[_Box],
                 *,
                 max_children: int = 16,
                 context: _Optional[_Context] = None) -> None:
        """
        Initializes tree from boxes.

        Time complexity:
            ``O(size * log size)``
        Memory complexity:
            ``O(size)``

        where ``size = len(boxes)``.

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Box = context.box_cls
        >>> boxes = [Box(-index, index, 0, index) for index in range(1, 11)]
        >>> tree = Tree(boxes)
        """
        if context is None:
            context = _get_context()
        self._boxes, self._context, self._max_children, self._root = (
            boxes, context, max_children,
            _create_root(boxes, max_children, context.merged_box,
                         context.box_point_squared_distance))

    __repr__ = _generate_repr(__init__)

    @property
    def boxes(self) -> _Sequence[_Box]:
        """
        Returns underlying boxes.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Box = context.box_cls
        >>> boxes = [Box(-index, index, 0, index) for index in range(1, 11)]
        >>> tree = Tree(boxes)
        >>> tree.boxes == boxes
        True
        """
        return self._boxes

    @property
    def context(self) -> _Context:
        """
        Returns context of the tree.

        Time complexity:
            ``O(1)``
        Memory complexity:
            ``O(1)``
        """
        return self._context

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
        >>> Box = context.box_cls
        >>> boxes = [Box(-index, index, 0, index) for index in range(1, 11)]
        >>> tree = Tree(boxes)
        >>> tree.max_children == 16
        True
        """
        return self._max_children

    def find_subsets(self, box: _Box) -> _List[_Box]:
        """
        Searches for boxes that lie inside the given box.

        Time complexity:
            ``O(max_children * log size + hits_count)``
        Memory complexity:
            ``O(max_children * log size + hits_count)``

        where ``size = len(self.boxes)``,
        ``max_children = self.max_children``,
        ``hits_count`` --- number of found boxes.

        :param box: input box.
        :returns: boxes that lie inside the input box.

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Box = context.box_cls
        >>> boxes = [Box(-index, index, 0, index) for index in range(1, 11)]
        >>> tree = Tree(boxes)
        >>> tree.find_subsets(Box(-1, 1, 0, 1)) == [Box(-1, 1, 0, 1)]
        True
        >>> (tree.find_subsets(Box(-2, 2, 0, 2))
        ...  == [Box(-1, 1, 0, 1), Box(-2, 2, 0, 2)])
        True
        >>> (tree.find_subsets(Box(-3, 3, 0, 3))
        ...  == [Box(-1, 1, 0, 1), Box(-2, 2, 0, 2), Box(-3, 3, 0, 3)])
        True
        """
        return [box for _, box in self._find_subsets_items(box)]

    def find_subsets_indices(self, box: _Box) -> _List[int]:
        """
        Searches for indices of boxes that lie inside the given box.

        Time complexity:
            ``O(max_children * log size + hits_count)``
        Memory complexity:
            ``O(max_children * log size + hits_count)``

        where ``size = len(self.boxes)``,
        ``max_children = self.max_children``,
        ``hits_count`` --- number of found indices.

        :param box: input box.
        :returns: indices of boxes that lie inside the input box.

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Box = context.box_cls
        >>> boxes = [Box(-index, index, 0, index) for index in range(1, 11)]
        >>> tree = Tree(boxes)
        >>> tree.find_subsets_indices(Box(-1, 1, 0, 1)) == [0]
        True
        >>> tree.find_subsets_indices(Box(-2, 2, 0, 2)) == [0, 1]
        True
        >>> tree.find_subsets_indices(Box(-3, 3, 0, 3)) == [0, 1, 2]
        True
        """
        return [index for index, _ in self._find_subsets_items(box)]

    def find_subsets_items(self, box: _Box) -> _List[Item]:
        """
        Searches for indices with boxes that lie inside the given box.

        Time complexity:
            ``O(max_children * log size + hits_count)``
        Memory complexity:
            ``O(max_children * log size + hits_count)``

        where ``size = len(self.boxes)``,
        ``max_children = self.max_children``,
        ``hits_count`` --- number of found indices with boxes.

        :param box: input box.
        :returns: indices with boxes that lie inside the input box.

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Box = context.box_cls
        >>> boxes = [Box(-index, index, 0, index) for index in range(1, 11)]
        >>> tree = Tree(boxes)
        >>> (tree.find_subsets_items(Box(-1, 1, 0, 1))
        ...  == [(0, Box(-1, 1, 0, 1))])
        True
        >>> (tree.find_subsets_items(Box(-2, 2, 0, 2))
        ...  == [(0, Box(-1, 1, 0, 1)), (1, Box(-2, 2, 0, 2))])
        True
        >>> (tree.find_subsets_items(Box(-3, 3, 0, 3))
        ...  == [(0, Box(-1, 1, 0, 1)), (1, Box(-2, 2, 0, 2)),
        ...      (2, Box(-3, 3, 0, 3))])
        True
        """
        return list(self._find_subsets_items(box))

    def find_supersets(self, box: _Box) -> _List[_Box]:
        """
        Searches for boxes that contain the given box.

        Time complexity:
            ``O(max_children * log size + hits_count)``
        Memory complexity:
            ``O(max_children * log size + hits_count)``

        where ``size = len(self.boxes)``,
        ``max_children = self.max_children``,
        ``hits_count`` --- number of found boxes.

        :param box: input box.
        :returns: boxes that contain the input box.

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Box = context.box_cls
        >>> boxes = [Box(-index, index, 0, index) for index in range(1, 11)]
        >>> tree = Tree(boxes)
        >>> tree.find_supersets(Box(-10, 10, 0, 10)) == [Box(-10, 10, 0, 10)]
        True
        >>> (tree.find_supersets(Box(-9, 9, 0, 9))
        ...  == [Box(-9, 9, 0, 9), Box(-10, 10, 0, 10)])
        True
        >>> (tree.find_supersets(Box(-8, 8, 0, 8))
        ...  == [Box(-8, 8, 0, 8), Box(-9, 9, 0, 9), Box(-10, 10, 0, 10)])
        True
        """
        return [box
                for _, box in self._find_supersets_items(box)]

    def find_supersets_indices(self, box: _Box) -> _List[int]:
        """
        Searches for indices of boxes that contain the given box.

        Time complexity:
            ``O(max_children * log size + hits_count)``
        Memory complexity:
            ``O(max_children * log size + hits_count)``

        where ``size = len(self.boxes)``,
        ``max_children = self.max_children``,
        ``hits_count`` --- number of found indices.

        :param box: input box.
        :returns: indices of boxes that contain the input box.

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Box = context.box_cls
        >>> boxes = [Box(-index, index, 0, index) for index in range(1, 11)]
        >>> tree = Tree(boxes)
        >>> tree.find_supersets_indices(Box(-10, 10, 0, 10)) == [9]
        True
        >>> tree.find_supersets_indices(Box(-9, 9, 0, 9)) == [8, 9]
        True
        >>> tree.find_supersets_indices(Box(-8, 8, 0, 8)) == [7, 8, 9]
        True
        """
        return [index for index, _ in self._find_supersets_items(box)]

    def find_supersets_items(self, box: _Box) -> _List[Item]:
        """
        Searches for indices with boxes
        that contain the given box.

        Time complexity:
            ``O(max_children * log size + hits_count)``
        Memory complexity:
            ``O(max_children * log size + hits_count)``

        where ``size = len(self.boxes)``,
        ``max_children = self.max_children``,
        ``hits_count`` --- number of found indices with boxes.

        :param box: input box.
        :returns: indices with boxes that contain the input box.

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Box = context.box_cls
        >>> boxes = [Box(-index, index, 0, index) for index in range(1, 11)]
        >>> tree = Tree(boxes)
        >>> (tree.find_supersets_items(Box(-10, 10, 0, 10))
        ...  == [(9, Box(-10, 10, 0, 10))])
        True
        >>> (tree.find_supersets_items(Box(-9, 9, 0, 9))
        ...  == [(8, Box(-9, 9, 0, 9)), (9, Box(-10, 10, 0, 10))])
        True
        >>> (tree.find_supersets_items(Box(-8, 8, 0, 8))
        ...  == [(7, Box(-8, 8, 0, 8)), (8, Box(-9, 9, 0, 9)),
        ...      (9, Box(-10, 10, 0, 10))])
        True
        """
        return list(self._find_supersets_items(box))

    def _find_subsets_items(self, box: _Box) -> _Iterator[Item]:
        yield from (enumerate(self._boxes)
                    if _box.is_subset_of(self._root.box, box)
                    else _find_node_box_subsets_items(self._root,
                                                      box))

    def _find_supersets_items(self, box: _Box) -> _Iterator[Item]:
        yield from _find_node_box_supersets_items(self._root, box)

    def n_nearest_indices(self, n: int, point: _Point) -> _Sequence[int]:
        """
        Searches for indices of boxes in the tree
        the nearest to the given point.

        Time complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(1)`` otherwise
        Memory complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(1)`` otherwise

        where ``size = len(self.boxes)``,
        ``max_children = self.max_children``.

        :param n: positive upper bound for number of result indices.
        :param point: input point.
        :returns:
            indices of boxes in the tree the nearest to the input point.

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Box, Point = context.box_cls, context.point_cls
        >>> boxes = [Box(-index, index, 0, index) for index in range(1, 11)]
        >>> tree = Tree(boxes)
        >>> tree.n_nearest_indices(2, Point(0, 0)) == [9, 8]
        True
        >>> (tree.n_nearest_indices(len(boxes), Point(0, 0))
        ...  == range(len(boxes)))
        True
        """
        return ([index for index, _ in self._n_nearest_items(n, point)]
                if n < len(self._boxes)
                else range(len(self._boxes)))

    def n_nearest_boxes(self, n: int, point: _Point) -> _Sequence[_Box]:
        """
        Searches for boxes in the tree the nearest to the given point.

        Time complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(1)`` otherwise
        Memory complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(1)`` otherwise

        where ``size = len(self.boxes)``,
        ``max_children = self.max_children``.

        :param n: positive upper bound for number of result boxes.
        :param point: input point.
        :returns: boxes in the tree the nearest to the input point.

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Box, Point = context.box_cls, context.point_cls
        >>> boxes = [Box(-index, index, 0, index) for index in range(1, 11)]
        >>> tree = Tree(boxes)
        >>> (tree.n_nearest_boxes(2, Point(0, 0))
        ...  == [Box(-10, 10, 0, 10), Box(-9, 9, 0, 9)])
        True
        >>> tree.n_nearest_boxes(len(boxes), Point(0, 0)) == boxes
        True
        """
        return ([box for _, box in self._n_nearest_items(n, point)]
                if n < len(self._boxes)
                else self._boxes)

    def n_nearest_items(self, n: int, point: _Point) -> _Sequence[Item]:
        """
        Searches for indices with boxes in the tree
        the nearest to the given point.

        Time complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(size)`` otherwise
        Memory complexity:
            ``O(n * max_children * log size)`` if ``n < size``,
            ``O(size)`` otherwise

        where ``size = len(self.boxes)``,
        ``max_children = self.max_children``.

        :param n:
            positive upper bound for number of result indices with boxes.
        :param point: input point.
        :returns:
            indices with boxes in the tree the nearest to the input point.

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Box, Point = context.box_cls, context.point_cls
        >>> boxes = [Box(-index, index, 0, index) for index in range(1, 11)]
        >>> tree = Tree(boxes)
        >>> (tree.n_nearest_items(2, Point(0, 0))
        ...  == [(9, Box(-10, 10, 0, 10)), (8, Box(-9, 9, 0, 9))])
        True
        >>> (tree.n_nearest_items(len(boxes), Point(0, 0))
        ...  == list(enumerate(boxes)))
        True
        """
        return list(self._n_nearest_items(n, point)
                    if n < len(self._boxes)
                    else enumerate(self._boxes))

    def nearest_index(self, point: _Point) -> int:
        """
        Searches for index of box in the tree
        the nearest to the given point.

        Time complexity:
            ``O(max_children * log size)``
        Memory complexity:
            ``O(max_children * log size)``

        where ``size = len(self.boxes)``,
        ``max_children = self.max_children``.

        :param point: input point.
        :returns: index of box in the tree the nearest to the input point.

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Box, Point = context.box_cls, context.point_cls
        >>> boxes = [Box(-index, index, 0, index) for index in range(1, 11)]
        >>> tree = Tree(boxes)
        >>> tree.nearest_index(Point(0, 0)) == 9
        True
        """
        result, _ = self.nearest_item(point)
        return result

    def nearest_box(self, point: _Point) -> _Box:
        """
        Searches for box in the tree the nearest to the given point.

        Time complexity:
            ``O(max_children * log size)``
        Memory complexity:
            ``O(max_children * log size)``

        where ``size = len(self.boxes)``,
        ``max_children = self.max_children``.

        :param point: input point.
        :returns: box in the tree the nearest to the input point.

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Box, Point = context.box_cls, context.point_cls
        >>> boxes = [Box(-index, index, 0, index) for index in range(1, 11)]
        >>> tree = Tree(boxes)
        >>> tree.nearest_box(Point(0, 0)) == Box(-10, 10, 0, 10)
        True
        """
        _, result = self.nearest_item(point)
        return result

    def nearest_item(self, point: _Point) -> Item:
        """
        Searches for index with box in the tree
        the nearest to the given point.

        Time complexity:
            ``O(max_children * log size)``
        Memory complexity:
            ``O(max_children * log size)``

        where ``size = len(self.boxes)``,
        ``max_children = self.max_children``.

        :param point: input point.
        :returns:
            index with box in the tree the nearest to the input point.

        >>> from ground.base import get_context
        >>> context = get_context()
        >>> Box, Point = context.box_cls, context.point_cls
        >>> boxes = [Box(-index, index, 0, index) for index in range(1, 11)]
        >>> tree = Tree(boxes)
        >>> tree.nearest_item(Point(0, 0)) == (9, Box(-10, 10, 0, 10))
        True
        >>> tree.nearest_item(Point(-10, 0)) == (9, Box(-10, 10, 0, 10))
        True
        >>> tree.nearest_item(Point(-10, 10)) == (9, Box(-10, 10, 0, 10))
        True
        >>> tree.nearest_item(Point(10, 0)) == (9, Box(-10, 10, 0, 10))
        True
        >>> tree.nearest_item(Point(10, 10)) == (9, Box(-10, 10, 0, 10))
        True
        """
        queue = [(0, 0, self._root)]
        while queue:
            _, _, node = _heappop(queue)
            for child in node.children:
                _heappush(queue,
                          (child.distance_to_point(point),
                           -child.index - 1 if child.is_leaf else child.index,
                           child))
            if queue and queue[0][1] < 0:
                _, _, node = _heappop(queue)
                return node.item

    def _n_nearest_items(self, n: int, point: _Point) -> _Iterator[Item]:
        queue = [(0, 0, self._root)]
        while n and queue:
            _, _, node = _heappop(queue)
            for child in node.children:
                _heappush(queue,
                          (child.distance_to_point(point),
                           -child.index - 1 if child.is_leaf else child.index,
                           child))
            while n and queue and queue[0][1] < 0:
                _, _, node = _heappop(queue)
                yield node.item
                n -= 1


def _create_root(boxes: _Sequence[_Box],
                 max_children: int,
                 boxes_merger: _Callable[[_Box, _Box], _Box],
                 metric: _Callable[[_Box, _Point], _Coordinate]) -> Node:
    nodes = [Node(index, box, None, metric)
             for index, box in enumerate(boxes)]
    root_box = _reduce(boxes_merger, boxes)
    leaves_count = len(nodes)
    if leaves_count <= max_children:
        # only one node, skip sorting and just fill the root box
        return Node(len(nodes), root_box, nodes, metric)
    else:
        def node_key(node: Node,
                     double_root_delta_x: _Coordinate
                     = 2 * (root_box.max_x - root_box.min_x),
                     double_root_delta_y: _Coordinate
                     = 2 * (root_box.max_y - root_box.min_y),
                     double_root_min_x: _Coordinate = 2 * root_box.min_x,
                     double_root_min_y: _Coordinate = 2 * root_box.min_y
                     ) -> int:
            box = node.box
            return _hilbert.index(_floor(_hilbert.MAX_COORDINATE
                                         * (box.min_x + box.max_x
                                            - double_root_min_x)
                                         / double_root_delta_x),
                                  _floor(_hilbert.MAX_COORDINATE
                                         * (box.min_y + box.max_y
                                            - double_root_min_y)
                                         / double_root_delta_y))

        nodes = sorted(nodes,
                       key=node_key)
        nodes_count = step = leaves_count
        levels_limits = [nodes_count]
        while True:
            step = _ceil_division(step, max_children)
            if step == 1:
                break
            nodes_count += step
            levels_limits.append(nodes_count)
        start = 0
        for level_limit in levels_limits:
            while start < level_limit:
                stop = min(start + max_children, level_limit)
                children = nodes[start:stop]
                nodes.append(Node(len(nodes),
                                  _reduce(boxes_merger,
                                          [child.box for child in children]),
                                  children, metric))
                start = stop
        return nodes[-1]


def _node_to_leaves(node: Node) -> _Iterator[Node]:
    if node.is_leaf:
        yield node
    else:
        for child in node.children:
            yield from _node_to_leaves(child)


def _find_node_box_subsets_items(node: Node, box: _Box) -> _Iterator[Item]:
    if _box.is_subset_of(node.box, box):
        for leaf in _node_to_leaves(node):
            yield leaf.item
    elif not node.is_leaf and _box.overlaps(box, node.box):
        for child in node.children:
            yield from _find_node_box_subsets_items(child, box)


def _find_node_box_supersets_items(node: Node, box: _Box) -> _Iterator[Item]:
    if _box.is_subset_of(box, node.box):
        if node.is_leaf:
            yield node.item
        else:
            for child in node.children:
                yield from _find_node_box_supersets_items(child, box)
