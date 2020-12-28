from typing import Tuple

from ground.hints import Box
from hypothesis import given

from locus.core.box import contains_point
from locus.kd import Tree
from tests.utils import (is_point)
from . import strategies


@given(strategies.trees_with_boxes)
def test_basic(tree_with_box: Tuple[Tree, Box]) -> None:
    tree, box = tree_with_box

    result = tree.find_box_points(box)

    assert isinstance(result, list)
    assert all(is_point(element) for element in result)


@given(strategies.trees_with_boxes)
def test_properties(tree_with_box: Tuple[Tree, Box]) -> None:
    tree, box = tree_with_box

    result = tree.find_box_points(box)

    assert all(point in tree.points for point in result)
    assert all(contains_point(box, point) for point in result)
    assert all(point in result
               for point in tree.points
               if contains_point(box, point))
