from heapq import nsmallest
from typing import Tuple

from ground.hints import (Scalar,
                          Segment)
from hypothesis import given

from locus.segmental import Tree
from tests.utils import (is_segment,
                         to_segments_distance)
from . import strategies


@given(strategies.trees_with_segments_and_sizes)
def test_basic(tree_with_segment_and_n: Tuple[Tree, Segment, int]) -> None:
    tree, segment, n = tree_with_segment_and_n

    result = tree.n_nearest_segments(n, segment)

    assert isinstance(result, (list, type(tree.segments)))
    assert all(is_segment(element) for element in result)


@given(strategies.trees_with_segments_and_sizes)
def test_properties(tree_with_segment_and_n: Tuple[Tree, Segment, int]
                    ) -> None:
    tree, segment, n = tree_with_segment_and_n

    result = tree.n_nearest_segments(n, segment)

    def to_segment_distance(tree_segment: Segment) -> Scalar:
        return to_segments_distance(tree_segment, segment)

    assert 0 < len(result) <= n
    assert all(segment in tree.segments for segment in result)
    assert (set(nsmallest(n, map(to_segment_distance, tree.segments)))
            == set(map(to_segment_distance, result)))
