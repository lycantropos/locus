from typing import Tuple

from hypothesis import given

from locus.core.interval import is_subset_of
from locus.hints import Interval
from locus.r import Tree
from . import strategies


@given(strategies.trees_with_intervals)
def test_basic(tree_with_interval: Tuple[Tree, Interval]) -> None:
    tree, interval = tree_with_interval

    result = tree.find_supersets_indices(interval)

    assert isinstance(result, list)
    assert all(isinstance(element, int) for element in result)


@given(strategies.trees)
def test_base_intervals(tree: Tree) -> None:
    assert all(index in tree.find_supersets_indices(interval)
               for index, interval in enumerate(tree.intervals))


@given(strategies.trees_with_intervals)
def test_properties(tree_with_interval: Tuple[Tree, Interval]) -> None:
    tree, interval = tree_with_interval

    result = tree.find_supersets_indices(interval)

    indices = range(len(tree.intervals))
    assert all(index in indices for index in result)
    assert all(is_subset_of(interval, tree.intervals[index])
               for index in result)
    assert all(index in result
               for index in indices
               if is_subset_of(interval, tree.intervals[index]))
