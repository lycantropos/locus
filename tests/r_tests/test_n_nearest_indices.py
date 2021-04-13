from heapq import nsmallest
from typing import Tuple

from ground.hints import (Coordinate,
                          Point)
from hypothesis import given

from locus.r import Tree
from tests.utils import to_box_point_distance
from . import strategies


@given(strategies.trees_with_points_and_sizes)
def test_basic(tree_with_point_and_n: Tuple[Tree, Point, int]) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_indices(n, point)

    assert isinstance(result, (list, range))
    assert all(isinstance(element, int) for element in result)


@given(strategies.trees_with_points_and_sizes)
def test_properties(tree_with_point_and_n: Tuple[Tree, Point, int]) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_indices(n, point)

    def to_point_distance(index: int) -> Coordinate:
        return to_box_point_distance(tree.boxes[index], point)

    indices = range(len(tree.boxes))
    assert 0 < len(result) <= n
    assert all(index in indices for index in result)
    assert (set(nsmallest(n, map(to_point_distance, indices)))
            == set(map(to_point_distance, result)))
