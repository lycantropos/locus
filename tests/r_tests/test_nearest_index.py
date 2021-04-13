from typing import Tuple

from ground.hints import (Coordinate,
                          Point)
from hypothesis import given

from locus.r import Tree
from tests.utils import to_box_point_distance
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
        return to_box_point_distance(tree.boxes[index], point)

    indices = range(len(tree.boxes))
    assert result in indices
    assert min(map(to_point_distance, indices)) == to_point_distance(result)
