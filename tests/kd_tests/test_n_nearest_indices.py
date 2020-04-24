from heapq import nsmallest
from typing import Tuple

from hypothesis import given

from locus.core.utils import squared_distance
from locus.hints import (Coordinate,
                         Point)
from locus.kd import Tree
from . import strategies


@given(strategies.trees_with_points_and_sizes)
def test_basic(tree_with_point_and_n: Tuple[Tree, Point, int]) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_indices(n, point)

    assert isinstance(result, list)
    assert all(isinstance(element, int) for element in result)


@given(strategies.trees_with_points_and_sizes)
def test_properties(tree_with_point_and_n: Tuple[Tree, Point, int]) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_indices(n, point)

    def to_point_distance(index: int) -> Coordinate:
        return squared_distance(tree.points[index], point)

    indices = range(len(tree.points))
    assert 0 < len(result) <= n
    assert all(point in indices for point in result)
    assert (set(nsmallest(n, map(to_point_distance, indices)))
            == set(map(to_point_distance, result)))
