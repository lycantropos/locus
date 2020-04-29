from heapq import nsmallest
from typing import Tuple

from hypothesis import given

from locus.core.interval import planar_distance_to_point
from locus.hints import (Coordinate,
                         Interval,
                         Point)
from locus.r import Tree
from tests.utils import is_interval
from . import strategies


@given(strategies.trees_with_points_and_sizes)
def test_basic(tree_with_point_and_n: Tuple[Tree, Point, int]) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_intervals(n, point)

    assert isinstance(result, (list, type(tree.intervals)))
    assert all(is_interval(element) for element in result)


@given(strategies.trees_with_points_and_sizes)
def test_properties(tree_with_point_and_n: Tuple[Tree, Point, int]) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_intervals(n, point)

    def to_point_distance(interval: Interval) -> Coordinate:
        return planar_distance_to_point(interval, point)

    assert 0 < len(result) <= n
    assert all(interval in tree.intervals for interval in result)
    assert (set(nsmallest(n, map(to_point_distance, tree.intervals)))
            == set(map(to_point_distance, result)))
