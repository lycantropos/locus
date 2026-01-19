from heapq import nsmallest

from ground.hints import Segment
from hypothesis import given

from locus._core.segmental import Item
from locus.segmental import Tree
from tests.hints import ScalarT
from tests.utils import is_segmental_item, to_segment_squared_distance

from . import strategies


@given(strategies.tree_with_segment_and_size_strategy)
def test_basic(
    tree_with_segment_and_n: tuple[Tree[ScalarT], Segment[ScalarT], int],
) -> None:
    tree, segment, n = tree_with_segment_and_n

    result = tree.n_nearest_items(n, segment)

    assert isinstance(result, list)
    assert all(is_segmental_item(element) for element in result)


@given(strategies.tree_with_segment_and_size_strategy)
def test_properties(
    tree_with_segment_and_n: tuple[Tree[ScalarT], Segment[ScalarT], int],
) -> None:
    tree, segment, n = tree_with_segment_and_n

    result = tree.n_nearest_items(n, segment)

    def to_segment_distance(item: Item[ScalarT], /) -> ScalarT:
        return to_segment_squared_distance(item[1], segment)

    items = list(enumerate(tree.segments))
    assert 0 < len(result) <= n
    assert all(item in items for item in result)
    assert set(nsmallest(n, map(to_segment_distance, items))) == set(
        map(to_segment_distance, result)
    )
