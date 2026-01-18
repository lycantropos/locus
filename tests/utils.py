from collections.abc import Callable, Iterable, Sequence
from functools import partial
from itertools import groupby
from math import ceil, log2
from operator import is_
from typing import Any, TypeVar, cast

from ground.context import get_context
from ground.hints import Box, Point, Segment
from hypothesis import strategies
from hypothesis.strategies import SearchStrategy

from locus import kd, r, segmental
from locus.core.box import is_subset_of
from locus.core.kd import NIL, Nil, Node as KdNode
from locus.core.r import AnyNode as AnyRNode, is_leaf as is_r_leaf
from locus.core.segmental import Node as SegmentalNode
from tests.hints import ScalarT

Strategy = SearchStrategy
context = get_context()


equivalence = is_


_T = TypeVar('_T')


def pack(function: Callable[..., _T], /) -> Callable[[Iterable[Any]], _T]:
    return partial(call, function)


def call(function: Callable[..., _T], args: Iterable[Any], /) -> _T:
    return function(*args)


def to_pairs(elements: Strategy[_T], /) -> Strategy[tuple[_T, _T]]:
    return strategies.tuples(elements, elements)


def is_kd_tree_balanced(tree: kd.Tree[ScalarT], /) -> bool:
    return tree._root is NIL or is_kd_node_balanced(tree._root)


def is_r_tree_balanced(tree: r.Tree[ScalarT], /) -> bool:
    return is_r_node_balanced(tree._root)


def is_segmental_tree_balanced(tree: segmental.Tree[ScalarT], /) -> bool:
    return is_segmental_node_balanced(tree._root)


def is_kd_tree_valid(tree: kd.Tree[ScalarT], /) -> bool:
    return tree._root is NIL or is_kd_node_valid(tree.points, tree._root)


def is_r_tree_valid(tree: r.Tree[ScalarT], /) -> bool:
    return is_r_node_valid(tree._root)


def is_segmental_tree_valid(tree: segmental.Tree[ScalarT], /) -> bool:
    return is_segmental_node_valid(tree._root)


def to_balanced_tree_height(size: int, max_children: int) -> int:
    size_log2 = size.bit_length() - 1
    return (
        size_log2
        if max_children == 2
        else ceil(size_log2 / log2(max_children))
    )


def to_kd_tree_height(tree: kd.Tree[ScalarT], /) -> int:
    return to_kd_node_height(tree._root)


def to_r_tree_height(tree: r.Tree[ScalarT], /) -> int:
    return to_r_node_height(tree._root)


def to_segmental_tree_height(tree: segmental.Tree[ScalarT], /) -> int:
    return to_segmental_node_height(tree._root)


def is_kd_node_balanced(node: KdNode[ScalarT], /) -> bool:
    return abs(
        to_kd_node_height(node.left) - to_kd_node_height(node.right)
    ) <= 1 and all(
        is_kd_node_balanced(child) for child in to_kd_node_children(node)
    )


def is_r_node_balanced(node: AnyRNode[ScalarT], /) -> bool:
    if is_r_leaf(node):
        return True
    children_heights = list(map(to_r_node_height, node.children))
    return max(children_heights) - min(children_heights) <= 1 and all(
        is_r_node_balanced(child) for child in node.children
    )


def is_segmental_node_balanced(node: SegmentalNode[ScalarT], /) -> bool:
    if node.is_leaf:
        return True
    assert node.children is not None, node
    children_heights = list(map(to_segmental_node_height, node.children))
    return max(children_heights) - min(children_heights) <= 1 and all(
        is_segmental_node_balanced(child) for child in node.children
    )


def is_kd_node_valid(
    points: Sequence[Point[ScalarT]], node: KdNode[ScalarT], /
) -> bool:
    hyperplane = node.projector(points[node.index])
    if node.left is not NIL and hyperplane < node.projector(
        points[node.left.index]
    ):
        return False
    if (
        node.right is not NIL
        and node.projector(points[node.right.index]) < hyperplane
    ):
        return False
    return all(
        is_kd_node_valid(points, child) for child in to_kd_node_children(node)
    )


def is_r_node_valid(node: AnyRNode[ScalarT], /) -> bool:
    return is_r_leaf(node) or (
        all(is_subset_of(child.box, node.box) for child in node.children)
        and all(is_r_node_valid(child) for child in node.children)
    )


def is_segmental_node_valid(node: SegmentalNode[ScalarT], /) -> bool:
    if node.is_leaf:
        return True
    assert node.children is not None, node
    return all(
        is_subset_of(child.box, node.box) for child in node.children
    ) and all(is_segmental_node_valid(child) for child in node.children)


def to_kd_node_height(node: KdNode[ScalarT] | Nil) -> int:
    if node is NIL:
        return -1
    return max(
        [1 + to_kd_node_height(child) for child in to_kd_node_children(node)],
        default=0,
    )


def to_r_node_height(node: AnyRNode[ScalarT], /) -> int:
    return (
        0
        if is_r_leaf(node)
        else max(1 + to_r_node_height(child) for child in node.children)
    )


def to_segmental_node_height(node: SegmentalNode[ScalarT], /) -> int:
    if node.is_leaf:
        return 0
    assert node.children is not None, node
    return max(1 + to_segmental_node_height(child) for child in node.children)


def to_kd_node_children(node: KdNode[ScalarT], /) -> Iterable[KdNode[ScalarT]]:
    if node.left is not NIL:
        yield node.left
    if node.right is not NIL:
        yield node.right


def is_kd_item(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], int)
        and isinstance(value[1], context.point_cls)
    )


def is_r_item(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], int)
        and value[0] >= 0
        and isinstance(value[1], context.box_cls)
    )


def is_segmental_item(value: Any, /) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], int)
        and value[0] >= 0
        and isinstance(value[1], context.segment_cls)
    )


def all_equal(iterable: Iterable[_T], /) -> bool:
    groups = groupby(iterable)
    return next(groups, True) and not next(groups, False)


def all_unique(iterable: Iterable[_T]) -> bool:
    seen: set[_T] = set()
    register = seen.add
    for element in iterable:
        if element in seen:
            return False
        register(element)
    return True


def identity(value: _T, /) -> _T:
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


def rot(size: int, x: int, y: int, rx: int, ry: int) -> tuple[int, int]:
    if not ry:
        if rx == 1:
            x, y = size - 1 - x, size - 1 - y
        x, y = y, x
    return x, y


def to_box_point_squared_distance(
    box: Box[ScalarT], point: Point[ScalarT], /
) -> ScalarT:
    return cast(ScalarT, context.box_point_squared_distance(box, point))


def to_point_squared_distance(
    first: Point[ScalarT], second: Point[ScalarT], /
) -> ScalarT:
    return cast(ScalarT, context.points_squared_distance(first, second))


def to_segment_point_squared_distance(
    segment: Segment[ScalarT], point: Point[ScalarT], /
) -> ScalarT:
    return cast(
        ScalarT, context.segment_point_squared_distance(segment, point)
    )


def to_segment_squared_distance(
    first: Segment[ScalarT], second: Segment[ScalarT], /
) -> ScalarT:
    return cast(ScalarT, context.segments_squared_distance(first, second))
