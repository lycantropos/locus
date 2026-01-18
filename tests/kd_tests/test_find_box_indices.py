from ground.hints import Box
from hypothesis import given

from locus.core.box import contains_point
from locus.kd import Tree
from tests.hints import ScalarT

from . import strategies


@given(strategies.tree_with_box_strategy)
def test_basic(tree_with_box: tuple[Tree[ScalarT], Box[ScalarT]]) -> None:
    tree, box = tree_with_box

    result = tree.find_box_indices(box)

    assert isinstance(result, list)
    assert all(isinstance(element, int) for element in result)


@given(strategies.tree_with_box_strategy)
def test_properties(tree_with_box: tuple[Tree[ScalarT], Box[ScalarT]]) -> None:
    tree, box = tree_with_box

    result = tree.find_box_indices(box)

    indices = range(len(tree.points))
    assert all(index in indices for index in result)
    assert all(contains_point(box, tree.points[index]) for index in result)
    assert all(
        index in result
        for index in indices
        if contains_point(box, tree.points[index])
    )
