from heapq import nsmallest
from typing import Tuple

from hypothesis import given

from locus.core.segment import distance_to
from locus.hints import (Coordinate,
                         Segment)
from locus.segmental import (Item,
                             Tree)
from tests.utils import is_segmental_item
from . import strategies


@given(strategies.trees_with_segments_and_sizes)
def test_basic(tree_with_segment_and_n: Tuple[Tree, Segment, int]) -> None:
    tree, segment, n = tree_with_segment_and_n

    result = tree.n_nearest_items(n, segment)

    assert isinstance(result, list)
    assert all(is_segmental_item(element) for element in result)


@given(strategies.trees_with_segments_and_sizes)
def test_properties(tree_with_segment_and_n: Tuple[Tree, Segment, int]
                    ) -> None:
    tree, segment, n = tree_with_segment_and_n

    result = tree.n_nearest_items(n, segment)

    def to_segment_distance(item: Item) -> Coordinate:
        return distance_to(item[1], segment)

    items = list(enumerate(tree.segments))
    assert 0 < len(result) <= n
    assert all(item in items for item in result)
    assert (set(nsmallest(n, map(to_segment_distance, items)))
            == set(map(to_segment_distance, result)))
