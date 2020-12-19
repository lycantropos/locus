from functools import partial
from typing import Tuple

from ground.hints import Point
from hypothesis import given

from locus.kd import Tree
from tests.utils import (all_unique,
                         equivalence,
                         is_point,
                         to_points_distance)
from . import strategies


@given(strategies.trees_with_points)
def test_basic(tree_with_point: Tuple[Tree, Point]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_point(point)

    assert is_point(result)


@given(strategies.trees)
def test_fixed_points(tree: Tree) -> None:
    assert all(tree.nearest_point(point) == point for point in tree.points)


@given(strategies.trees)
def test_uniqueness_criteria(tree: Tree) -> None:
    assert equivalence(all(tree.nearest_point(point) is point
                           for point in tree.points),
                       all_unique(tree.points))


@given(strategies.trees_with_points)
def test_properties(tree_with_point: Tuple[Tree, Point]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_point(point)

    to_point_distance = partial(to_points_distance, point)
    assert result in tree.points
    assert (min(map(to_point_distance, tree.points))
            == to_point_distance(result))
