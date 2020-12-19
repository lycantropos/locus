from heapq import nsmallest
from typing import Tuple

from ground.hints import (Coordinate,
                          Segment)
from hypothesis import given

from locus.segmental import Tree
from tests.utils import to_segments_distance
from . import strategies


@given(strategies.trees_with_segments_and_sizes)
def test_basic(tree_with_segment_and_n: Tuple[Tree, Segment, int]) -> None:
    tree, segment, n = tree_with_segment_and_n

    result = tree.n_nearest_indices(n, segment)

    assert isinstance(result, (list, range))
    assert all(isinstance(element, int) for element in result)


@given(strategies.trees_with_segments_and_sizes)
def test_properties(tree_with_segment_and_n: Tuple[Tree, Segment, int]
                    ) -> None:
    tree, segment, n = tree_with_segment_and_n

    result = tree.n_nearest_indices(n, segment)

    def to_segment_distance(index: int) -> Coordinate:
        return to_segments_distance(tree.segments[index], segment)

    indices = range(len(tree.segments))
    assert 0 < len(result) <= n
    assert all(index in indices for index in result)
    assert (set(nsmallest(n, map(to_segment_distance, indices)))
            == set(map(to_segment_distance, result)))
