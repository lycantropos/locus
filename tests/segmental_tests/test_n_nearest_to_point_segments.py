from heapq import nsmallest
from typing import Tuple

from hypothesis import given

from locus.core.segment import distance_to_point
from locus.hints import (Coordinate,
                         Point,
                         Segment)
from locus.segmental import Tree
from tests.utils import is_segment
from . import strategies


@given(strategies.trees_with_points_and_sizes)
def test_basic(tree_with_point_and_n: Tuple[Tree, Point, int]) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_to_point_segments(n, point)

    assert isinstance(result, (list, type(tree.segments)))
    assert all(is_segment(element) for element in result)


@given(strategies.trees_with_points_and_sizes)
def test_properties(tree_with_point_and_n: Tuple[Tree, Point, int]) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_to_point_segments(n, point)

    def to_point_distance(segment: Segment) -> Coordinate:
        return distance_to_point(segment, point)

    assert 0 < len(result) <= n
    assert all(segment in tree.segments for segment in result)
    assert (set(nsmallest(n, map(to_point_distance, tree.segments)))
            == set(map(to_point_distance, result)))
