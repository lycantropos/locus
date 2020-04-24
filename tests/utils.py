from itertools import groupby
from typing import (Any,
                    Iterable,
                    Tuple,
                    TypeVar,
                    Union)

from hypothesis import strategies
from hypothesis.strategies import SearchStrategy

from locus.hints import Coordinate
from locus.kd import (NIL,
                      Node,
                      Tree)

Domain = TypeVar('Domain')
Range = TypeVar('Range')
Strategy = SearchStrategy


def to_homogeneous_tuples(elements: Strategy[Domain],
                          *,
                          size: int) -> Strategy[Tuple[Domain, ...]]:
    return (strategies.lists(elements,
                             min_size=size,
                             max_size=size)
            .map(tuple))


def is_tree_balanced(tree: Tree) -> bool:
    return is_node_balanced(tree.root)


def is_tree_valid(tree: Tree) -> bool:
    return is_node_valid(tree.root)


def to_balanced_tree_height(size: int) -> int:
    return size.bit_length() - 1


def to_tree_height(tree: Tree) -> int:
    return to_node_height(tree.root)


def is_node_balanced(node: Node) -> bool:
    if abs(to_node_height(node.left) - to_node_height(node.right)) > 1:
        return False
    return all(is_node_balanced(child) for child in to_node_children(node))


def is_node_valid(node: Node) -> bool:
    if (node.left is not NIL
            and node.point[node.axis] < node.left.point[node.axis]):
        return False
    if (node.right is not NIL
            and node.point[node.axis] > node.right.point[node.axis]):
        return False
    return all(is_node_valid(child) for child in to_node_children(node))


def to_node_height(node: Union[Node, NIL]) -> int:
    if node is NIL:
        return -1
    return max([1 + to_node_height(child) for child in to_node_children(node)],
               default=0)


def to_node_children(node: Node) -> Iterable[Node]:
    if node.left is not NIL:
        yield node.left
    if node.right is not NIL:
        yield node.right


def is_point(value: Any) -> bool:
    return (isinstance(value, tuple)
            and len(value) > 0
            and all_equal(map(type, value)) == 1
            and all(isinstance(sub_element, Coordinate)
                    for sub_element in value))


def all_equal(iterable: Iterable[Domain]) -> bool:
    groups = groupby(iterable)
    return next(groups, True) and not next(groups, False)
