from typing import Tuple

from ground.hints import (Coordinate,
                          Point)
from hypothesis import given

from locus.core.interval import distance_to_point
from locus.hints import Interval
from locus.r import Tree
from tests.utils import is_interval
from . import strategies


@given(strategies.trees_with_points)
def test_basic(tree_with_point: Tuple[Tree, Point]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_interval(point)

    assert is_interval(result)


@given(strategies.trees_with_points)
def test_properties(tree_with_point: Tuple[Tree, Point]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_interval(point)

    def to_point_distance(interval: Interval) -> Coordinate:
        return distance_to_point(interval, point)

    assert result in tree.intervals
    assert (min(map(to_point_distance, tree.intervals))
            == to_point_distance(result))
