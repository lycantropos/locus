from ground.hints import Box
from hypothesis import given

from locus.core.box import contains_point
from locus.kd import Tree
from tests.hints import ScalarT
from tests.utils import context

from . import strategies


@given(strategies.tree_with_box_strategy)
def test_basic(tree_with_box: tuple[Tree[ScalarT], Box[ScalarT]]) -> None:
    tree, box = tree_with_box

    result = tree.find_box_points(box)

    assert isinstance(result, list)
    assert all(isinstance(element, context.point_cls) for element in result)


@given(strategies.tree_with_box_strategy)
def test_properties(tree_with_box: tuple[Tree[ScalarT], Box[ScalarT]]) -> None:
    tree, box = tree_with_box

    result = tree.find_box_points(box)

    assert all(point in tree.points for point in result)
    assert all(contains_point(box, point) for point in result)
    assert all(
        point in result for point in tree.points if contains_point(box, point)
    )
