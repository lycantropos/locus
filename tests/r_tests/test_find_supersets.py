from typing import Tuple

from hypothesis import given

from locus.core.interval import is_subset_of
from locus.hints import Interval
from locus.r import Tree
from tests.utils import is_interval
from . import strategies


@given(strategies.trees_with_intervals)
def test_basic(tree_with_interval: Tuple[Tree, Interval]) -> None:
    tree, interval = tree_with_interval

    result = tree.find_supersets(interval)

    assert isinstance(result, list)
    assert all(is_interval(element) for element in result)


@given(strategies.trees)
def test_base_intervals(tree: Tree) -> None:
    assert all(interval in tree.find_supersets(interval)
               for interval in tree.intervals)


@given(strategies.trees_with_intervals)
def test_properties(tree_with_interval: Tuple[Tree, Interval]) -> None:
    tree, interval = tree_with_interval

    result = tree.find_supersets(interval)

    assert all(interval in tree.intervals for interval in result)
    assert all(is_subset_of(interval, result_interval)
               for result_interval in result)
    assert all(tree_interval in result
               for tree_interval in tree.intervals
               if is_subset_of(interval, tree_interval))
