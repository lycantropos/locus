from typing import Tuple

from hypothesis import given

from locus.core.segment import distance_to_point
from locus.hints import (Coordinate,
                         Point,
                         Segment)
from locus.segmental import Tree
from tests.utils import is_segment
from . import strategies


@given(strategies.trees_with_points)
def test_basic(tree_with_point: Tuple[Tree, Point]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_to_point_segment(point)

    assert is_segment(result)


@given(strategies.trees_with_points)
def test_properties(tree_with_point: Tuple[Tree, Point]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_to_point_segment(point)

    def to_point_distance(segment: Segment) -> Coordinate:
        return distance_to_point(segment, point)

    assert result in tree.segments
    assert (min(map(to_point_distance, tree.segments))
            == to_point_distance(result))
