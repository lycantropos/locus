from functools import partial
from typing import Tuple

from ground.hints import (Coordinate,
                          Point)
from hypothesis import given

from locus.kd import Tree
from tests.utils import (all_equal,
                         is_point,
                         to_points_distance)
from . import strategies


@given(strategies.trees_with_balls)
def test_basic(tree_with_ball: Tuple[Tree, Point, Coordinate]) -> None:
    tree, center, radius = tree_with_ball

    result = tree.find_ball_points(center, radius)

    assert isinstance(result, list)
    assert all(is_point(element) for element in result)


@given(strategies.trees_with_points)
def test_zero_ball(tree_with_center: Tuple[Tree, Point]) -> None:
    tree, center = tree_with_center

    result = tree.find_ball_points(center, 0)

    assert not result or set(result) == {center}


@given(strategies.trees_with_balls)
def test_properties(tree_with_ball: Tuple[Tree, Point, Coordinate]) -> None:
    tree, center, radius = tree_with_ball

    result = tree.find_ball_points(center, radius)

    to_center_distance = partial(to_points_distance, center)
    assert sum(center == point for point in tree.points) <= len(result)
    assert all_equal(map(len, result))
    assert all(point in tree.points for point in result)
    assert all(to_center_distance(point) <= radius
               for point in result)
    assert all(point in result
               for point in tree.points
               if to_center_distance(point) <= radius)
