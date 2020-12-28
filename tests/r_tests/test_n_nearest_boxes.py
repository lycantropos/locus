from heapq import nsmallest
from typing import Tuple

from ground.hints import (Box,
                          Coordinate,
                          Point)
from hypothesis import given

from locus.core.box import distance_to_point
from locus.r import Tree
from tests.utils import is_box
from . import strategies


@given(strategies.trees_with_points_and_sizes)
def test_basic(tree_with_point_and_n: Tuple[Tree, Point, int]) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_boxes(n, point)

    assert isinstance(result, (list, type(tree.boxes)))
    assert all(is_box(element) for element in result)


@given(strategies.trees_with_points_and_sizes)
def test_properties(tree_with_point_and_n: Tuple[Tree, Point, int]) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_boxes(n, point)

    def to_point_distance(box: Box) -> Coordinate:
        return distance_to_point(box, point)

    assert 0 < len(result) <= n
    assert all(box in tree.boxes for box in result)
    assert (set(nsmallest(n, map(to_point_distance, tree.boxes)))
            == set(map(to_point_distance, result)))
