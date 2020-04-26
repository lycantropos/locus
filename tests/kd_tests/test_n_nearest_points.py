from functools import partial
from heapq import nsmallest
from typing import Tuple

from hypothesis import given

from locus.core.utils import planar_distance
from locus.hints import Point
from locus.kd import Tree
from tests.utils import (all_equal,
                         is_point)
from . import strategies


@given(strategies.trees_with_points_and_sizes)
def test_basic(tree_with_point_and_n: Tuple[Tree, Point, int]) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_points(n, point)

    assert isinstance(result, (list, type(tree.points)))
    assert all(is_point(element) for element in result)


@given(strategies.trees_with_points_and_sizes)
def test_properties(tree_with_point_and_n: Tuple[Tree, Point, int]) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_points(n, point)

    to_point_distance = partial(planar_distance, point)
    assert 0 < len(result) <= n
    assert all_equal(map(len, result))
    assert all(point in tree.points for point in result)
    assert (set(nsmallest(n, map(to_point_distance, tree.points)))
            == set(map(to_point_distance, result)))