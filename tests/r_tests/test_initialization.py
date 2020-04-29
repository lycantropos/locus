from typing import List

from hypothesis import given

from locus.hints import Interval
from locus.r import Tree
from . import strategies


@given(strategies.intervals_lists, strategies.max_children_counts)
def test_basic(intervals: List[Interval], max_children: int) -> None:
    result = Tree(intervals,
                  max_children=max_children)

    assert result.intervals == intervals
    assert result.max_children == max_children
