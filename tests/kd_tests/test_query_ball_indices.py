from typing import Tuple

from hypothesis import given

from locus.core.utils import squared_distance
from locus.hints import (Coordinate,
                         Point)
from locus.kd import Tree
from . import strategies


@given(strategies.trees_with_balls)
def test_basic(tree_with_ball: Tuple[Tree, Point, Coordinate]) -> None:
    tree, center, radius = tree_with_ball

    result = tree.query_ball_indices(center, radius)

    assert isinstance(result, list)
    assert all(isinstance(element, int) for element in result)


@given(strategies.trees_with_points)
def test_zero_ball(tree_with_center: Tuple[Tree, Point]) -> None:
    tree, center = tree_with_center

    result = tree.query_ball_indices(center, 0)

    assert not result or {tree.points[index] for index in result} == {center}


@given(strategies.trees_with_balls)
def test_properties(tree_with_ball: Tuple[Tree, Point, Coordinate]) -> None:
    tree, center, radius = tree_with_ball

    result = tree.query_ball_indices(center, radius)

    def to_center_distance(index: int) -> Coordinate:
        return squared_distance(tree.points[index], center)

    indices = range(len(tree.points))
    assert sum(center == point for point in tree) <= len(result)
    assert all(index in indices for index in result)
    assert all(to_center_distance(point) <= radius * radius
               for point in result)
    assert all(index in result
               for index in indices
               if to_center_distance(index) <= radius * radius)
