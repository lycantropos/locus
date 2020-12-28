from typing import Tuple

from ground.hints import Box
from hypothesis import given

from locus.core.box import contains_point
from locus.kd import Tree
from . import strategies


@given(strategies.trees_with_boxes)
def test_basic(tree_with_box: Tuple[Tree, Box]) -> None:
    tree, box = tree_with_box

    result = tree.find_box_indices(box)

    assert isinstance(result, list)
    assert all(isinstance(element, int) for element in result)


@given(strategies.trees_with_boxes)
def test_properties(tree_with_box: Tuple[Tree, Box]) -> None:
    tree, box = tree_with_box

    result = tree.find_box_indices(box)

    indices = range(len(tree.points))
    assert all(index in indices for index in result)
    assert all(contains_point(box, tree.points[index])
               for index in result)
    assert all(index in result
               for index in indices
               if contains_point(box, tree.points[index]))
