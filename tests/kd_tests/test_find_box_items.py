from ground.hints import Box
from hypothesis import given

from locus._core.box import contains_point
from locus.kd import Tree
from tests.hints import ScalarT
from tests.utils import is_kd_item

from .strategies import tree_with_box_strategy


@given(tree_with_box_strategy)
def test_basic(tree_with_box: tuple[Tree[ScalarT], Box[ScalarT]]) -> None:
    tree, box = tree_with_box

    result = tree.find_box_items(box)

    assert isinstance(result, list)
    assert all(is_kd_item(element) for element in result)


@given(tree_with_box_strategy)
def test_properties(tree_with_box: tuple[Tree[ScalarT], Box[ScalarT]]) -> None:
    tree, box = tree_with_box

    result = tree.find_box_items(box)

    items = list(enumerate(tree.points))
    assert all(item in items for item in result)
    assert all(contains_point(box, point) for _, point in result)
    assert all(
        item in result for item in items if contains_point(box, item[1])
    )
