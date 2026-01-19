from ground.hints import Box
from hypothesis import given

from locus.r import Tree
from tests.hints import ScalarT
from tests.utils import (
    is_r_tree_balanced,
    is_r_tree_valid,
    to_balanced_tree_height,
    to_r_tree_height,
)

from . import strategies


@given(strategies.box_list_strategy, strategies.max_child_count_strategy)
def test_basic(boxes: list[Box[ScalarT]], max_children: int) -> None:
    result = Tree(boxes, max_children=max_children)

    assert result.boxes == boxes
    assert result.max_children == max_children


@given(strategies.box_list_strategy, strategies.max_child_count_strategy)
def test_properties(boxes: list[Box[ScalarT]], max_children: int) -> None:
    result = Tree(boxes, max_children=max_children)

    assert is_r_tree_valid(result)
    assert is_r_tree_balanced(result)
    assert to_r_tree_height(result) >= to_balanced_tree_height(
        len(boxes), max_children
    )
