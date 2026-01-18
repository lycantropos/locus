from ground.hints import Box
from hypothesis import given

from locus.core.box import is_subset_of
from locus.r import Tree
from tests.hints import ScalarT
from tests.utils import is_r_item

from . import strategies


@given(strategies.trees_with_boxes)
def test_basic(tree_with_box: tuple[Tree[ScalarT], Box[ScalarT]]) -> None:
    tree, box = tree_with_box

    result = tree.find_subsets_items(box)

    assert isinstance(result, list)
    assert all(is_r_item(element) for element in result)


@given(strategies.trees)
def test_base_boxes(tree: Tree[ScalarT]) -> None:
    assert all(
        (index, box) in tree.find_subsets_items(box)
        for index, box in enumerate(tree.boxes)
    )


@given(strategies.trees_with_boxes)
def test_properties(tree_with_box: tuple[Tree[ScalarT], Box[ScalarT]]) -> None:
    tree, box = tree_with_box

    result = tree.find_subsets_items(box)

    items = list(enumerate(tree.boxes))
    assert all(item in items for item in result)
    assert all(is_subset_of(result_box, box) for _, result_box in result)
    assert all(item in result for item in items if is_subset_of(item[1], box))
