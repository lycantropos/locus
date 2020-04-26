from typing import Tuple

from hypothesis import given

from locus.core.utils import point_in_interval
from locus.hints import Interval
from locus.kd import Tree
from tests.utils import (all_equal,
                         is_point)
from . import strategies


@given(strategies.trees_with_intervals)
def test_basic(tree_with_interval: Tuple[Tree, Interval]) -> None:
    tree, interval = tree_with_interval

    result = tree.query_interval_points(interval)

    assert isinstance(result, list)
    assert all(is_point(element) for element in result)


@given(strategies.trees_with_intervals)
def test_properties(tree_with_interval: Tuple[Tree, Interval]) -> None:
    tree, interval = tree_with_interval

    result = tree.query_interval_points(interval)

    assert all_equal(map(len, result))
    assert all(point in tree.points for point in result)
    assert all(point_in_interval(point, interval)
               for point in result)
    assert all(point in result
               for point in tree.points
               if point_in_interval(point, interval))
