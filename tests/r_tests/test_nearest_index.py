from typing import Tuple

from ground.hints import (Coordinate,
                          Point)
from hypothesis import given

from locus.core.interval import distance_to_point
from locus.r import Tree
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
        return distance_to_point(tree.intervals[index], point)

    indices = range(len(tree.intervals))
    assert result in indices
    assert min(map(to_point_distance, indices)) == to_point_distance(result)
