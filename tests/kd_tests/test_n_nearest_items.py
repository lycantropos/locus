from heapq import nsmallest
from typing import Tuple

from ground.hints import (Coordinate,
                          Point)
from hypothesis import given

from locus.kd import (Item,
                      Tree)
from tests.utils import (is_kd_item,
                         to_points_distance)
from . import strategies


@given(strategies.trees_with_points_and_sizes)
def test_basic(tree_with_point_and_n: Tuple[Tree, Point, int]) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_items(n, point)

    assert isinstance(result, list)
    assert all(is_kd_item(element) for element in result)


@given(strategies.trees_with_points_and_sizes)
def test_properties(tree_with_point_and_n: Tuple[Tree, Point, int]) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_items(n, point)

    def to_point_distance(item: Item) -> Coordinate:
        return to_points_distance(item[1], point)

    items = list(enumerate(tree.points))
    assert 0 < len(result) <= n
    assert all(point in items for point in result)
    assert (set(nsmallest(n, map(to_point_distance, items)))
            == set(map(to_point_distance, result)))
