from ground.hints import Box, Point
from hypothesis import given

from locus.r import Tree
from tests.hints import ScalarT
from tests.utils import context, to_box_point_squared_distance

from . import strategies


@given(strategies.tree_with_point_strategy)
def test_basic(tree_with_point: tuple[Tree[ScalarT], Point[ScalarT]]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_box(point)

    assert isinstance(result, context.box_cls)


@given(strategies.tree_with_point_strategy)
def test_properties(
    tree_with_point: tuple[Tree[ScalarT], Point[ScalarT]],
) -> None:
    tree, point = tree_with_point

    result = tree.nearest_box(point)

    def to_point_distance(box: Box[ScalarT]) -> ScalarT:
        return to_box_point_squared_distance(box, point)

    assert result in tree.boxes
    assert min(map(to_point_distance, tree.boxes)) == to_point_distance(result)
