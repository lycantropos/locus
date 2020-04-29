from typing import Tuple

from hypothesis import given

from locus.core.interval import contains_point
from locus.hints import Interval
from locus.kd import Tree
from . import strategies


@given(strategies.trees_with_intervals)
def test_basic(tree_with_interval: Tuple[Tree, Interval]) -> None:
    tree, interval = tree_with_interval

    result = tree.find_interval_indices(interval)

    assert isinstance(result, list)
    assert all(isinstance(element, int) for element in result)


@given(strategies.trees_with_intervals)
def test_properties(tree_with_interval: Tuple[Tree, Interval]) -> None:
    tree, interval = tree_with_interval

    result = tree.find_interval_indices(interval)

    indices = range(len(tree.points))
    assert all(index in indices for index in result)
    assert all(contains_point(interval, tree.points[index])
               for index in result)
    assert all(index in result
               for index in indices
               if contains_point(interval, tree.points[index]))
