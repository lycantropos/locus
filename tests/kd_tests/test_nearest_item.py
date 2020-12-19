from typing import Tuple

from ground.hints import (Coordinate,
                          Point)
from hypothesis import given

from locus.kd import (Item,
                      Tree)
from tests.utils import (all_unique,
                         equivalence,
                         is_kd_item,
                         to_points_distance)
from . import strategies


@given(strategies.trees_with_points)
def test_basic(tree_with_point: Tuple[Tree, Point]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_item(point)

    assert is_kd_item(result)


@given(strategies.trees)
def test_uniqueness_criteria(tree: Tree) -> None:
    assert equivalence(all(tree.nearest_item(point) == (index, point)
                           for index, point in enumerate(tree.points)),
                       all_unique(tree.points))


@given(strategies.trees_with_points)
def test_properties(tree_with_point: Tuple[Tree, Point]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_item(point)

    def to_point_distance(item: Item) -> Coordinate:
        return to_points_distance(item[1], point)

    items = list(enumerate(tree.points))
    assert result in items
    assert min(map(to_point_distance, items)) == to_point_distance(result)
