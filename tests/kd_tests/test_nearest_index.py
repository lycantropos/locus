from typing import Tuple

from ground.hints import (Coordinate,
                          Point)
from hypothesis import given

from locus.kd import Tree
from tests.utils import (all_unique,
                         equivalence,
                         to_points_distance)
from . import strategies


@given(strategies.trees_with_points)
def test_basic(tree_with_point: Tuple[Tree, Point]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_index(point)

    assert isinstance(result, int)


@given(strategies.trees)
def test_uniqueness_criteria(tree: Tree) -> None:
    assert equivalence(all(tree.nearest_index(point) == index
                           for index, point in enumerate(tree.points)),
                       all_unique(tree.points))


@given(strategies.trees_with_points)
def test_properties(tree_with_point: Tuple[Tree, Point]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_index(point)

    def to_point_distance(index: int) -> Coordinate:
        return to_points_distance(tree.points[index], point)

    indices = range(len(tree.points))
    assert result in indices
    assert min(map(to_point_distance, indices)) == to_point_distance(result)
