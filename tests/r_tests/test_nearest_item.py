from ground.hints import Point
from hypothesis import given

from locus._core.r import Item
from locus.r import Tree
from tests.hints import ScalarT
from tests.utils import is_r_item, to_box_point_squared_distance

from . import strategies


@given(strategies.tree_with_point_strategy)
def test_basic(tree_with_point: tuple[Tree[ScalarT], Point[ScalarT]]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_item(point)

    assert is_r_item(result)


@given(strategies.tree_with_point_strategy)
def test_properties(
    tree_with_point: tuple[Tree[ScalarT], Point[ScalarT]],
) -> None:
    tree, point = tree_with_point

    result = tree.nearest_item(point)

    def to_point_distance(item: Item[ScalarT], /) -> ScalarT:
        return to_box_point_squared_distance(item[1], point)

    items = list(enumerate(tree.boxes))
    assert result in items
    assert min(map(to_point_distance, items)) == to_point_distance(result)
