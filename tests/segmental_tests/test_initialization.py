from typing import List

from ground.hints import Segment
from hypothesis import given

from locus.segmental import Tree
from tests.utils import (is_r_tree_balanced,
                         is_r_tree_valid,
                         to_balanced_tree_height,
                         to_r_tree_height)
from . import strategies


@given(strategies.segments_lists, strategies.max_children_counts)
def test_basic(segments: List[Segment], max_children: int) -> None:
    result = Tree(segments,
                  max_children=max_children)

    assert result.segments == segments
    assert result.max_children == max_children


@given(strategies.segments_lists, strategies.max_children_counts)
def test_properties(segments: List[Segment], max_children: int) -> None:
    result = Tree(segments,
                  max_children=max_children)

    assert is_r_tree_valid(result)
    assert is_r_tree_balanced(result)
    assert to_r_tree_height(result) >= to_balanced_tree_height(len(segments),
                                                               max_children)
