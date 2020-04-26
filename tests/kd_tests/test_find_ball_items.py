from typing import Tuple

from hypothesis import given

from locus.core.utils import planar_distance
from locus.hints import (Coordinate,
                         Point)
from locus.kd import (Item,
                      Tree)
from tests.utils import is_item
from . import strategies


@given(strategies.trees_with_balls)
def test_basic(tree_with_ball: Tuple[Tree, Point, Coordinate]) -> None:
    tree, center, radius = tree_with_ball

    result = tree.find_ball_items(center, radius)

    assert isinstance(result, list)
    assert all(is_item(element) for element in result)


@given(strategies.trees_with_points)
def test_zero_ball(tree_with_center: Tuple[Tree, Point]) -> None:
    tree, center = tree_with_center

    result = tree.find_ball_items(center, 0)

    assert not result or {point for _, point in result} == {center}


@given(strategies.trees_with_balls)
def test_properties(tree_with_ball: Tuple[Tree, Point, Coordinate]) -> None:
    tree, center, radius = tree_with_ball

    result = tree.find_ball_items(center, radius)

    def to_center_distance(item: Item) -> Coordinate:
        return planar_distance(item[1], center)

    items = list(enumerate(tree.points))
    assert sum(center == point for point in tree.points) <= len(result)
    assert all(index in items for index in result)
    assert all(to_center_distance(point) <= radius
               for point in result)
    assert all(item in result
               for item in items
               if to_center_distance(item) <= radius)
