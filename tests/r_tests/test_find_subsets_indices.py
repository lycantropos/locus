from typing import Tuple

from ground.hints import Box
from hypothesis import given

from locus.core.box import is_subset_of
from locus.r import Tree
from . import strategies


@given(strategies.trees_with_boxes)
def test_basic(tree_with_box: Tuple[Tree, Box]) -> None:
    tree, box = tree_with_box

    result = tree.find_subsets_indices(box)

    assert isinstance(result, list)
    assert all(isinstance(element, int) for element in result)


@given(strategies.trees)
def test_base_boxes(tree: Tree) -> None:
    assert all(index in tree.find_subsets_indices(box)
               for index, box in enumerate(tree.boxes))


@given(strategies.trees_with_boxes)
def test_properties(tree_with_box: Tuple[Tree, Box]) -> None:
    tree, box = tree_with_box

    result = tree.find_subsets_indices(box)

    indices = range(len(tree.boxes))
    assert all(index in indices for index in result)
    assert all(is_subset_of(tree.boxes[index], box)
               for index in result)
    assert all(index in result
               for index in indices
               if is_subset_of(tree.boxes[index], box))
