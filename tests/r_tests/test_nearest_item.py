from typing import Tuple

from ground.hints import (Point,
                          Scalar)
from hypothesis import given

from locus.core.r import Item
from locus.r import Tree
from tests.utils import (is_r_item,
                         to_box_point_distance)
from . import strategies


@given(strategies.trees_with_points)
def test_basic(tree_with_point: Tuple[Tree, Point]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_item(point)

    assert is_r_item(result)


@given(strategies.trees_with_points)
def test_properties(tree_with_point: Tuple[Tree, Point]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_item(point)

    def to_point_distance(item: Item) -> Scalar:
        return to_box_point_distance(item[1], point)

    items = list(enumerate(tree.boxes))
    assert result in items
    assert min(map(to_point_distance, items)) == to_point_distance(result)
