from heapq import nsmallest
from typing import Tuple

from ground.hints import (Coordinate,
                          Point)
from hypothesis import given

from locus.r import (Item,
                     Tree)
from tests.utils import (is_r_item,
                         to_box_point_distance)
from . import strategies


@given(strategies.trees_with_points_and_sizes)
def test_basic(tree_with_point_and_n: Tuple[Tree, Point, int]) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_items(n, point)

    assert isinstance(result, list)
    assert all(is_r_item(element) for element in result)


@given(strategies.trees_with_points_and_sizes)
def test_properties(tree_with_point_and_n: Tuple[Tree, Point, int]) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_items(n, point)

    def to_point_distance(item: Item) -> Coordinate:
        return to_box_point_distance(item[1], point)

    items = list(enumerate(tree.boxes))
    assert 0 < len(result) <= n
    assert all(item in items for item in result)
    assert (set(nsmallest(n, map(to_point_distance, items)))
            == set(map(to_point_distance, result)))
