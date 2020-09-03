from typing import Tuple

from hypothesis import given

from locus.core.segment import distance_to_point
from locus.hints import (Coordinate,
                         Point)
from locus.segmental import (Item,
                             Tree)
from tests.utils import is_segmental_item
from . import strategies


@given(strategies.trees_with_points)
def test_basic(tree_with_point: Tuple[Tree, Point]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_to_point_item(point)

    assert is_segmental_item(result)


@given(strategies.trees_with_points)
def test_properties(tree_with_point: Tuple[Tree, Point]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_to_point_item(point)

    def to_point_distance(item: Item) -> Coordinate:
        return distance_to_point(item[1], point)

    items = list(enumerate(tree.segments))
    assert result in items
    assert min(map(to_point_distance, items)) == to_point_distance(result)