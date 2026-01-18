from ground.hints import Point
from hypothesis import given

from locus.core.segmental import Item
from locus.segmental import Tree
from tests.hints import ScalarT
from tests.utils import is_segmental_item, to_segment_point_squared_distance

from . import strategies


@given(strategies.tree_with_point_strategy)
def test_basic(tree_with_point: tuple[Tree[ScalarT], Point[ScalarT]]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_to_point_item(point)

    assert is_segmental_item(result)


@given(strategies.tree_with_point_strategy)
def test_properties(
    tree_with_point: tuple[Tree[ScalarT], Point[ScalarT]],
) -> None:
    tree, point = tree_with_point

    result = tree.nearest_to_point_item(point)

    def to_point_distance(item: Item[ScalarT], /) -> ScalarT:
        return to_segment_point_squared_distance(item[1], point)

    items = list(enumerate(tree.segments))
    assert result in items
    assert min(map(to_point_distance, items)) == to_point_distance(result)
