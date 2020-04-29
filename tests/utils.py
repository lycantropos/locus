from itertools import groupby
from typing import (Any,
                    Iterable,
                    Sequence,
                    Tuple,
                    TypeVar,
                    Union)

from hypothesis import strategies
from hypothesis.strategies import SearchStrategy
from math import (ceil,
                  log)

from locus import (kd,
                   r)
from locus.core.interval import is_subset_of
from locus.hints import (Coordinate,
                         Point)

Domain = TypeVar('Domain')
Range = TypeVar('Range')
Strategy = SearchStrategy


def equivalence(left_statement: bool, right_statement: bool) -> bool:
    return left_statement is right_statement


def to_homogeneous_tuples(elements: Strategy[Domain],
                          *,
                          size: int) -> Strategy[Tuple[Domain, ...]]:
    return (strategies.lists(elements,
                             min_size=size,
                             max_size=size)
            .map(tuple))


def is_kd_tree_balanced(tree: kd.Tree) -> bool:
    return is_kd_node_balanced(tree._root)


def is_r_tree_balanced(tree: r.Tree) -> bool:
    return is_r_node_balanced(tree._root)


def is_kd_tree_valid(tree: kd.Tree) -> bool:
    return is_kd_node_valid(tree.points, tree._root)


def is_r_tree_valid(tree: r.Tree) -> bool:
    return is_r_node_valid(tree._root)


def to_balanced_tree_height(size: int, max_children: int) -> int:
    size_log2 = size.bit_length() - 1
    return (size_log2
            if max_children == 2
            else ceil(size_log2 / log(max_children, 2)))


def to_kd_tree_height(tree: kd.Tree) -> int:
    return to_kd_node_height(tree._root)


def to_r_tree_height(tree: r.Tree) -> int:
    return to_r_node_height(tree._root)


def is_kd_node_balanced(node: kd.Node) -> bool:
    return (abs(to_kd_node_height(node.left)
                - to_kd_node_height(node.right)) <= 1
            and all(is_kd_node_balanced(child)
                    for child in to_kd_node_children(node)))


def is_r_node_balanced(node: r.Node) -> bool:
    if node.is_leaf:
        return True
    else:
        children_heights = list(map(to_r_node_height, node.children))
        return (max(children_heights) - min(children_heights) <= 1
                and all(is_r_node_balanced(child) for child in node.children))


def is_kd_node_valid(points: Sequence[Point], node: kd.Node) -> bool:
    hyperplane = points[node.index][node.axis]
    if (node.left is not kd.NIL
            and hyperplane < points[node.left.index][node.axis]):
        return False
    if (node.right is not kd.NIL
            and hyperplane > points[node.right.index][node.axis]):
        return False
    return all(is_kd_node_valid(points, child)
               for child in to_kd_node_children(node))


def is_r_node_valid(node: r.Node) -> bool:
    if node.is_leaf:
        return True
    else:
        return (all(is_subset_of(child.interval, node.interval)
                    for child in node.children)
                and all(is_r_node_valid(child) for child in node.children))


def to_kd_node_height(node: Union[kd.Node, kd.NIL]) -> int:
    if node is kd.NIL:
        return -1
    return max([1 + to_kd_node_height(child)
                for child in to_kd_node_children(node)],
               default=0)


def to_r_node_height(node: r.Node) -> int:
    return (0
            if node.is_leaf
            else max(1 + to_r_node_height(child) for child in node.children))


def to_kd_node_children(node: kd.Node) -> Iterable[kd.Node]:
    if node.left is not kd.NIL:
        yield node.left
    if node.right is not kd.NIL:
        yield node.right


def is_kd_item(value: Any) -> bool:
    return (isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], int)
            and is_point(value[1]))


def is_point(value: Any) -> bool:
    return (isinstance(value, tuple)
            and len(value) > 0
            and all_equal(map(type, value))
            and all(isinstance(sub_element, Coordinate)
                    for sub_element in value))


def all_equal(iterable: Iterable[Domain]) -> bool:
    groups = groupby(iterable)
    return next(groups, True) and not next(groups, False)


def all_unique(iterable: Iterable[Domain]) -> bool:
    seen = set()
    register = seen.add
    for element in iterable:
        if element in seen:
            return False
        else:
            register(element)
    return True


def identity(value: Domain) -> Domain:
    return value


def to_hilbert_index_complete(size: int, x: int, y: int) -> int:
    result = 0
    step = size // 2
    while step > 0:
        rx = (x & step) > 0
        ry = (y & step) > 0
        result += step * step * ((3 * rx) ^ ry)
        x, y = rot(size, x, y, rx, ry)
        step //= 2
    return result


def rot(size: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
    if not ry:
        if rx == 1:
            x, y = size - 1 - x, size - 1 - y
        x, y = y, x
    return x, y
