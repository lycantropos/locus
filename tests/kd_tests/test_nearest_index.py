from typing import Tuple

from hypothesis import given

from locus.core.utils import squared_distance
from locus.hints import (Coordinate,
                         Point)
from locus.kd import Tree
from . import strategies


@given(strategies.trees_with_points)
def test_basic(tree_with_point: Tuple[Tree, Point]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_index(point)

    assert isinstance(result, int)


@given(strategies.trees_with_points)
def test_properties(tree_with_point: Tuple[Tree, Point]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_index(point)

    def to_point_distance(index: int) -> Coordinate:
        return squared_distance(tree.points[index], point)

    indices = range(len(tree.points))
    assert result in indices
    assert min(map(to_point_distance, indices)) == to_point_distance(result)
