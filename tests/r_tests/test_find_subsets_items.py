from typing import Tuple

from hypothesis import given

from locus.core.interval import is_subset_of
from locus.hints import Interval
from locus.r import Tree
from tests.utils import is_r_item
from . import strategies


@given(strategies.trees_with_intervals)
def test_basic(tree_with_interval: Tuple[Tree, Interval]) -> None:
    tree, interval = tree_with_interval

    result = tree.find_subsets_items(interval)

    assert isinstance(result, list)
    assert all(is_r_item(element) for element in result)


@given(strategies.trees)
def test_base_intervals(tree: Tree) -> None:
    assert all((index, interval) in tree.find_subsets_items(interval)
               for index, interval in enumerate(tree.intervals))


@given(strategies.trees_with_intervals)
def test_properties(tree_with_interval: Tuple[Tree, Interval]) -> None:
    tree, interval = tree_with_interval

    result = tree.find_subsets_items(interval)

    items = list(enumerate(tree.intervals))
    assert all(item in items for item in result)
    assert all(is_subset_of(result_interval, interval)
               for _, result_interval in result)
    assert all(item in result
               for item in items
               if is_subset_of(item[1], interval))
