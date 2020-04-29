from typing import List

from hypothesis import given

from locus.hints import Interval
from locus.r import Tree
from tests.utils import (is_r_tree_balanced,
                         is_r_tree_valid,
                         to_balanced_tree_height,
                         to_r_tree_height)
from . import strategies


@given(strategies.intervals_lists, strategies.max_children_counts)
def test_basic(intervals: List[Interval], max_children: int) -> None:
    result = Tree(intervals,
                  max_children=max_children)

    assert result.intervals == intervals
    assert result.max_children == max_children


@given(strategies.intervals_lists, strategies.max_children_counts)
def test_properties(intervals: List[Interval], max_children: int) -> None:
    result = Tree(intervals,
                  max_children=max_children)

    assert is_r_tree_valid(result)
    assert is_r_tree_balanced(result)
    assert to_r_tree_height(result) >= to_balanced_tree_height(len(intervals),
                                                               max_children)
