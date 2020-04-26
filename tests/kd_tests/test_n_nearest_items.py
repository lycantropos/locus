from heapq import nsmallest
from typing import Tuple

from hypothesis import given

from locus.core.utils import planar_distance
from locus.hints import (Coordinate,
                         Point)
from locus.kd import (Item,
                      Tree)
from tests.utils import is_item
from . import strategies


@given(strategies.trees_with_points_and_sizes)
def test_basic(tree_with_point_and_n: Tuple[Tree, Point, int]) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_items(n, point)

    assert isinstance(result, list)
    assert all(is_item(element) for element in result)


@given(strategies.trees_with_points_and_sizes)
def test_properties(tree_with_point_and_n: Tuple[Tree, Point, int]) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_items(n, point)

    def to_point_distance(item: Item) -> Coordinate:
        return planar_distance(item[1], point)

    items = list(enumerate(tree.points))
    assert 0 < len(result) <= n
    assert all(point in items for point in result)
    assert (set(nsmallest(n, map(to_point_distance, items)))
            == set(map(to_point_distance, result)))