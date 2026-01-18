from ground.hints import Box
from hypothesis import given

from locus.core.box import is_subset_of
from locus.r import Tree
from tests.hints import ScalarT
from tests.utils import context

from . import strategies


@given(strategies.tree_with_box_strategy)
def test_basic(tree_with_box: tuple[Tree[ScalarT], Box[ScalarT]]) -> None:
    tree, box = tree_with_box

    result = tree.find_subsets(box)

    assert isinstance(result, list)
    assert all(isinstance(element, context.box_cls) for element in result)


@given(strategies.trees)
def test_base_boxes(tree: Tree[ScalarT]) -> None:
    assert all(box in tree.find_subsets(box) for box in tree.boxes)


@given(strategies.tree_with_box_strategy)
def test_properties(tree_with_box: tuple[Tree[ScalarT], Box[ScalarT]]) -> None:
    tree, box = tree_with_box

    result = tree.find_subsets(box)

    assert all(box in tree.boxes for box in result)
    assert all(is_subset_of(result_box, box) for result_box in result)
    assert all(
        tree_box in result
        for tree_box in tree.boxes
        if is_subset_of(tree_box, box)
    )
