from typing import Tuple

from ground.hints import Box
from hypothesis import given

from locus.core.box import contains_point
from locus.kd import Tree
from tests.utils import is_kd_item
from . import strategies


@given(strategies.trees_with_boxes)
def test_basic(tree_with_box: Tuple[Tree, Box]) -> None:
    tree, box = tree_with_box

    result = tree.find_box_items(box)

    assert isinstance(result, list)
    assert all(is_kd_item(element) for element in result)


@given(strategies.trees_with_boxes)
def test_properties(tree_with_box: Tuple[Tree, Box]) -> None:
    tree, box = tree_with_box

    result = tree.find_box_items(box)

    items = list(enumerate(tree.points))
    assert all(item in items for item in result)
    assert all(contains_point(box, point)
               for _, point in result)
    assert all(item in result
               for item in items
               if contains_point(box, item[1]))
